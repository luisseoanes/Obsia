#!/usr/bin/env python3
"""
Script para generar embeddings y base de datos vectorial optimizada para móvil
Procesa PDFs médicos, crea chunks inteligentes y genera índice FAISS cuantizado

CORRECCIONES APLICADAS:
  - Chunk size reducido a 60-120 palabras (óptimo para RAG en móvil con modelos 3B)
  - Chunk IDs globales (antes se reiniciaban por cada PDF)
  - Overlap entre chunks (~20%) para preservar contexto en bordes
  - has_clinical_content más estricto (separa términos genéricos de específicos)
  - model_card_vars corregido (AttributeError en sentence-transformers reciente)
  - nprobe proporcional al número de clusters (antes hardcodeado a 10)
  - Modelo MiniLM (~120MB) óptimo para móvil (mpnet era 1.1GB)
  - pdfplumber reemplaza PyPDF2 (mejor soporte para doble columna)
  - pymupdf como extractor secundario robusto
  - OCR con surya-ocr como fallback para PDFs escaneados (sin dependencias del sistema)
  - Limpieza de artefactos de extracción PDF (guiones, saltos de línea, etc.)

DEPENDENCIAS:
  pip install sentence-transformers faiss-cpu pdfplumber pymupdf pillow tqdm surya-ocr
  (surya-ocr descarga modelos ~1GB la primera vez que se usa)
"""

import json
import re
import io
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# VALIDACIÓN DE DEPENDENCIAS
# ------------------------------------------------------------------

MISSING = []

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    MISSING.append("sentence-transformers")

try:
    import faiss
except ImportError:
    MISSING.append("faiss-cpu")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    MISSING.append("pdfplumber")

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    MISSING.append("pymupdf")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    MISSING.append("pillow")

try:
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det_model
    from surya.model.detection.processor import load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
    HAS_SURYA = True
except ImportError:
    HAS_SURYA = False
    # Opcional — fallback para PDFs escaneados
    # Instala con: pip install surya-ocr

try:
    from tqdm import tqdm
except ImportError:
    MISSING.append("tqdm")

if MISSING:
    print("❌ Faltan dependencias. Ejecuta:")
    print(f"   pip install {' '.join(MISSING)}")
    exit(1)

if not HAS_PDFPLUMBER and not HAS_PYMUPDF:
    print("❌ Se necesita al menos pdfplumber o pymupdf.")
    exit(1)


# ------------------------------------------------------------------
# CLASE PRINCIPAL
# ------------------------------------------------------------------

class MedicalRAGGenerator:
    """Generador de base de datos RAG optimizada para contenido clínico obstétrico"""

    # Términos clínicos ESPECÍFICOS — validan que el chunk tiene contenido médico real
    SPECIFIC_CLINICAL_TERMS = {
        # Condiciones obstétricas
        'preeclampsia', 'eclampsia', 'hellp', 'embarazo', 'gestación', 'parto',
        'cesárea', 'hemorragia', 'postparto', 'posparto', 'prenatal', 'perinatal',
        'neonatal', 'placenta', 'útero', 'cervix', 'dilatación', 'contracciones',
        'trabajo de parto', 'alumbramiento', 'expulsivo', 'puerperio', 'gestante',
        'obstétric',

        # Signos y síntomas
        'hipertensión', 'proteinuria', 'edema', 'cefalea', 'convulsiones', 'oliguria',
        'trombocitopenia', 'hemólisis', 'presión arterial', 'tensión arterial',
        'atonía', 'acretismo', 'ruptura uterina', 'distocia', 'taquicardia',
        'bradicardia', 'hipotensión', 'sangrado',

        # Medicamentos y tratamientos
        'sulfato de magnesio', 'nifedipina', 'metildopa', 'hidralazina',
        'oxitocina', 'misoprostol', 'betametasona', 'dexametasona', 'labetalol',
        'ácido tranexámico', 'carbetocina', 'metilergonovina', 'ergometrina',
        'heparina', 'enoxaparina', 'ampicilina', 'gentamicina', 'clindamicina',
        'norepinefrina', 'vasopresores', 'uterotónico', 'tocolítico',
        'gluconato de calcio', 'noradrenalina',

        # Procedimientos
        'monitoreo fetal', 'cardiotocografía', 'ecografía', 'ultrasonido',
        'amniocentesis', 'cordocentesis', 'inducción', 'maduración cervical',
        'episiotomía', 'fórceps', 'ventosa', 'versión externa',
        'legrado', 'curetaje', 'laparotomía', 'histerectomía',
        'taponamiento', 'balón de bakri', 'sutura de b-lynch',

        # Anatomía obstétrica
        'feto', 'embrión', 'cordón umbilical', 'líquido amniótico', 'membranas',
        'cuello uterino', 'pelvis', 'uterino', 'placentario', 'amniótico',
        'trofoblasto', 'corion', 'endometrio', 'miometrio',

        # Escalas clínicas
        'apgar', 'bishop', 'índice de choque', 'shock index', 'sofa', 'qsofa',
        'silverman', 'ballard', 'capurro',

        # Valores clínicos
        'semanas de gestación', 'trimestre', 'mg/kg', 'ui/hora', 'mmhg',
        'g/dl', 'plaquetas', 'fibrinógeno', 'creatinina', 'transaminasas',
        'lactato', 'hematocrito', 'hemoglobina',

        # Emergencias obstétricas
        'código rojo', 'hemorragia masiva', 'transfusión', 'reanimación',
        'choque hipovolémico', 'coagulación intravascular', 'cid',
        'sepsis obstétrica', 'embolia de líquido amniótico',
        'corioamnionitis', 'endometritis', 'aborto séptico',

        # Cuidado crítico
        'morbilidad materna', 'mortalidad materna', 'near miss', 'uaco',
        'unidad de alta dependencia', 'cuidado intensivo obstétrico',
    }

    # Términos GENÉRICOS — no cuentan para validación
    GENERIC_TERMS = {
        'diagnóstico', 'manejo', 'tratamiento', 'guía', 'protocolo',
        'algoritmo', 'recomendación', 'evaluación', 'seguimiento',
        'indicaciones', 'contraindicaciones', 'clasificación', 'definición',
        'criterios', 'monitoreo', 'prevención', 'complicaciones',
    }

    # Secciones no clínicas a filtrar
    NON_CLINICAL_PATTERNS = [
        r'agradecimientos?',
        r'\bautores?\b',
        r'^índice\b',
        r'tabla de contenido',
        r'referencias bibliográficas',
        r'\bbibliografía\b',
        r'\banexos?\b',
        r'\bprólogo\b',
        r'^presentación\b',
        r'metodología de búsqueda',
        r'conflicto de inter[eé]s',
        r'declaración de',
        r'comité editorial',
        r'©\s*\d{4}',
        r'\bisbn\b',
        r'\bissn\b',
        r'todos los derechos reservados',
        r'página \d+ de \d+',
        r'^\d+\s*$',
        r'^(figura|tabla|gráfico)\s+\d+',
        r'fuentes? de financiaci[oó]n',
    ]

    # Tipos de sección clínica
    CLINICAL_SECTION_PATTERNS = [
        (r'definici[oó]n', 'Definición'),
        (r'criterios? diagn[oó]sticos?', 'Criterios Diagnósticos'),
        (r'diagn[oó]stico', 'Diagnóstico'),
        (r'clasificaci[oó]n', 'Clasificación'),
        (r'manejo', 'Manejo'),
        (r'tratamiento', 'Tratamiento'),
        (r'indicaciones?', 'Indicaciones'),
        (r'contraindicaciones?', 'Contraindicaciones'),
        (r'dosis|dosificaci[oó]n', 'Dosis'),
        (r'algoritmo', 'Algoritmo'),
        (r'protocolo', 'Protocolo'),
        (r'signos? de alarma', 'Signos de Alarma'),
        (r'complicaciones?', 'Complicaciones'),
        (r'factores? de riesgo', 'Factores de Riesgo'),
        (r'prevenci[oó]n', 'Prevención'),
        (r'seguimiento', 'Seguimiento'),
        (r'monitoreo|monitorizaci[oó]n', 'Monitoreo'),
        (r'evaluaci[oó]n', 'Evaluación'),
        (r'epidemiolog[ií]a', 'Epidemiología'),
        (r'fisiopatolog[ií]a', 'Fisiopatología'),
        (r'reanimaci[oó]n', 'Reanimación'),
        (r'emergencia', 'Emergencia'),
    ]

    def __init__(
        self,
        knowledge_base_dir: str,
        output_dir: str,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        min_words: int = 60,
        max_words: int = 120,
        overlap_sentences: int = 2,
        min_text_words_for_ocr: int = 80,
    ):
        """
        Args:
            knowledge_base_dir: Directorio con PDFs
            output_dir: Directorio de salida
            model_name: Modelo de embeddings (MiniLM ~120MB, óptimo para móvil)
            min_words: Mínimo de palabras por chunk
            max_words: Máximo de palabras por chunk
            overlap_sentences: Oraciones del chunk anterior a conservar (overlap)
            min_text_words_for_ocr: Umbral para activar OCR fallback con surya
        """
        self.kb_dir = Path(knowledge_base_dir)
        self.output_dir = Path(output_dir)
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences
        self.min_text_words_for_ocr = min_text_words_for_ocr

        print(f"🔧 Cargando modelo: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Modelo cargado. Dimensión: {self.embedding_dim}")

        print(f"📄 pdfplumber:  {'✅' if HAS_PDFPLUMBER else '❌ no disponible'}")
        print(f"📄 pymupdf:     {'✅' if HAS_PYMUPDF else '❌ no disponible'}")
        print(f"🔍 OCR surya:   {'✅ listo' if HAS_SURYA else '⚠️  no disponible — pip install surya-ocr'}")

        # Carga diferida de modelos surya (solo si se van a necesitar)
        self._surya_models = None

    # ------------------------------------------------------------------
    # EXTRACCIÓN EN CASCADA
    # ------------------------------------------------------------------

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Cascada de extracción:
          1. pdfplumber  → mejor para doble columna
          2. pymupdf     → fallback robusto
          3. surya-ocr   → fallback para PDFs escaneados (imágenes)
        """
        text = ""

        # 1. pdfplumber
        if HAS_PDFPLUMBER:
            text = self._extract_pdfplumber(pdf_path)

        # 2. pymupdf si pdfplumber dio poco texto
        if HAS_PYMUPDF and len(text.split()) < self.min_text_words_for_ocr:
            text_fitz = self._extract_pymupdf(pdf_path)
            if len(text_fitz.split()) > len(text.split()):
                text = text_fitz

        # 3. surya-ocr si aún hay poco texto (PDF escaneado)
        if HAS_SURYA and len(text.split()) < self.min_text_words_for_ocr:
            print(f"   🔍 Texto insuficiente ({len(text.split())}w) → surya OCR en {pdf_path.name}...")
            text_ocr = self._extract_ocr_surya(pdf_path)
            if len(text_ocr.split()) > len(text.split()):
                text = text_ocr

        return text

    def _extract_pdfplumber(self, pdf_path: Path) -> str:
        """
        pdfplumber con use_text_flow=True.
        Respeta el flujo natural del texto en documentos de doble columna,
        evitando la mezcla de columnas que sufre PyPDF2.
        """
        try:
            pages_text = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    words = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False,
                        use_text_flow=True,
                    )
                    if words:
                        page_text = " ".join(w["text"] for w in words)
                    else:
                        page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

                    if page_text.strip():
                        pages_text.append(page_text.strip())

            return "\n\n".join(pages_text)

        except Exception:
            return ""

    def _extract_pymupdf(self, pdf_path: Path) -> str:
        """pymupdf (fitz) — alternativa robusta para la mayoría de PDFs."""
        try:
            doc = fitz.open(str(pdf_path))
            pages_text = []
            for page in doc:
                text = page.get_text("text")
                if text.strip():
                    pages_text.append(text.strip())
            doc.close()
            return "\n\n".join(pages_text)
        except Exception:
            return ""

    def _extract_ocr_surya(self, pdf_path: Path) -> str:
        """
        OCR con surya-ocr — no requiere instalación del sistema.
        Los modelos se descargan automáticamente la primera vez (~1GB).
        Soporta español nativamente y maneja bien tablas y doble columna.

        Estrategia de carga diferida: los modelos surya son pesados (~1GB),
        por eso se cargan solo la primera vez que se necesitan y se reutilizan
        para todos los PDFs restantes del mismo procesamiento.
        """
        if not HAS_SURYA or not HAS_PYMUPDF:
            return ""

        try:
            # Carga diferida — solo la primera vez que se necesita OCR
            if self._surya_models is None:
                print("   📥 Cargando modelos surya-ocr (primera vez, puede tardar)...")
                self._surya_models = {
                    "det_model":     load_det_model(),
                    "det_processor": load_det_processor(),
                    "rec_model":     load_rec_model(),
                    "rec_processor": load_rec_processor(),
                }
                print("   ✅ Modelos surya cargados")

            det_model     = self._surya_models["det_model"]
            det_processor = self._surya_models["det_processor"]
            rec_model     = self._surya_models["rec_model"]
            rec_processor = self._surya_models["rec_processor"]

            doc = fitz.open(str(pdf_path))
            pages_text = []

            for page_num, page in enumerate(doc):
                # Renderizar página como imagen a 150 DPI (balance calidad/velocidad)
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

                # surya-ocr: pasar imagen + idioma objetivo
                predictions = run_ocr(
                    [img],
                    [["es"]],           # español
                    det_model,
                    det_processor,
                    rec_model,
                    rec_processor,
                )

                # Extraer texto de todas las líneas detectadas en la página
                if predictions and predictions[0].text_lines:
                    page_text = " ".join(
                        line.text for line in predictions[0].text_lines
                        if line.text.strip()
                    )
                    if page_text.strip():
                        pages_text.append(page_text.strip())

            doc.close()
            return "\n\n".join(pages_text)

        except Exception as e:
            print(f"   ⚠️  surya OCR fallido en {pdf_path.name}: {e}")
            return ""

    # ------------------------------------------------------------------
    # FILTROS Y VALIDACIONES
    # ------------------------------------------------------------------

    def is_non_clinical_section(self, text: str) -> bool:
        """Detecta si una sección es no clínica."""
        text_lower = text.lower().strip()
        return any(re.search(p, text_lower) for p in self.NON_CLINICAL_PATTERNS)

    def has_clinical_content(self, text: str) -> bool:
        """
        Valida contenido clínico con términos específicos.
        Se requieren mínimo 2 términos específicos obstétricos.
        Los términos genéricos (diagnóstico, manejo...) no cuentan.
        """
        text_lower = text.lower()
        specific_count = sum(1 for t in self.SPECIFIC_CLINICAL_TERMS if t in text_lower)
        return specific_count >= 2

    def detect_section_type(self, text: str) -> str:
        """Detecta el tipo de sección clínica."""
        text_lower = text.lower()
        for pattern, label in self.CLINICAL_SECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return label
        return "Contenido Clínico"

    def extract_clinical_topic(self, text: str) -> str:
        """Extrae el tema clínico principal del chunk."""
        text_lower = text.lower()

        topics = {
            'Preeclampsia':        ['preeclampsia', 'eclampsia', 'hellp', 'síndrome hipertensivo'],
            'Hemorragia':          ['hemorragia', 'sangrado', 'pérdida sanguínea', 'atonía', 'acretismo', 'hpp'],
            'Parto':               ['trabajo de parto', 'alumbramiento', 'expulsivo', 'dilatación'],
            'Cesárea':             ['cesárea', 'operación cesárea'],
            'Sepsis':              ['sepsis', 'infección', 'corioamnionitis', 'endometritis', 'choque séptico'],
            'Hipertensión':        ['hipertensión', 'presión arterial', 'tensión arterial', 'crisis hipertensiva'],
            'Diabetes Gestacional':['diabetes gestacional', 'diabetes en el embarazo', 'glucosa gestacional'],
            'Parto Pretérmino':    ['parto prematuro', 'pretérmino', 'amenaza de parto', 'tocolisis'],
            'Tromboembolismo':     ['tromboembolismo', 'trombosis', 'heparina', 'enoxaparina', 'tep', 'tvp'],
            'Monitoreo Fetal':     ['monitoreo fetal', 'cardiotocografía', 'fcf', 'bienestar fetal', 'nsst'],
            'Control Prenatal':    ['control prenatal', 'cuidado prenatal', 'atención prenatal'],
            'Infección / ITS':     ['vih', 'sífilis', 'toxoplasma', 'rubeola', 'hepatitis b', 'ets', 'its'],
            'Neonatología':        ['neonato', 'recién nacido', 'apgar', 'reanimación neonatal'],
            'Embarazo Ectópico':   ['embarazo ectópico', 'tubárico', 'ectópico'],
            'RCIU':                ['rciu', 'restricción del crecimiento', 'crecimiento intrauterino'],
            'Cuidado Crítico':     ['uci', 'cuidado intensivo', 'morbilidad materna', 'near miss', 'uaco'],
            'Distocia de Hombros': ['distocia de hombros', 'maniobra de mcroberts', 'maniobra de woods'],
        }

        for topic, keywords in topics.items():
            if any(kw in text_lower for kw in keywords):
                return topic

        return "Obstetricia General"

    # ------------------------------------------------------------------
    # CHUNKING CON OVERLAP
    # ------------------------------------------------------------------

    def intelligent_clinical_chunking(
        self,
        text: str,
        source: str,
        start_id: int = 0,
    ) -> List[Dict]:
        """
        Chunking inteligente con overlap entre chunks consecutivos.
        - IDs globales para unicidad entre PDFs
        - Filtra secciones no clínicas
        - Valida presencia de términos médicos específicos
        - Preserva contexto en bordes mediante overlap de N oraciones
        """
        chunks = []
        sections = re.split(r'\n\s*\n', text)
        chunk_id = start_id

        current_sentences: List[str] = []
        current_words: int = 0
        overlap_buffer: List[str] = []

        def flush_chunk() -> bool:
            nonlocal chunk_id
            full_sentences = overlap_buffer + current_sentences
            text_to_save = " ".join(full_sentences).strip()
            text_to_save = re.sub(r'\s+', ' ', text_to_save).strip()
            word_count = len(text_to_save.split())

            if word_count >= self.min_words and self.has_clinical_content(text_to_save):
                chunks.append(self._create_chunk(text_to_save, source, chunk_id))
                chunk_id += 1
                return True
            return False

        for section in sections:
            section = section.strip()
            if not section or self.is_non_clinical_section(section):
                continue

            section = self._clean_pdf_artifacts(section)
            if not section:
                continue

            sentences = re.split(r'(?<=[.!?;:])\s+', section)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence.split()) < 3:
                    continue

                sentence_words = len(sentence.split())

                if (current_words + sentence_words > self.max_words
                        and current_words >= self.min_words):
                    flush_chunk()
                    overlap_buffer = (
                        current_sentences[-self.overlap_sentences:]
                        if len(current_sentences) >= self.overlap_sentences
                        else current_sentences[:]
                    )
                    current_sentences = [sentence]
                    current_words = sentence_words
                else:
                    current_sentences.append(sentence)
                    current_words += sentence_words

        # Último chunk pendiente
        if current_sentences:
            flush_chunk()

        return chunks

    def _clean_pdf_artifacts(self, text: str) -> str:
        """
        Limpia artefactos comunes de extracción PDF:
        - Guiones de separación de sílabas al final de línea
        - Saltos de línea dentro de oraciones
        - Espacios múltiples
        - Números de página sueltos
        """
        # Unir palabras cortadas con guión (e.g. "hemorra-\ngia" → "hemorragia")
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

        # Saltos de línea simples → espacio (preserva párrafos dobles)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # Espacios múltiples
        text = re.sub(r'[ \t]+', ' ', text)

        # Números de página sueltos al final de línea
        text = re.sub(r'\b\d{1,3}\b\s*$', '', text, flags=re.MULTILINE)

        # Eliminar líneas muy cortas (probables headers/footers)
        lines = [l for l in text.split('\n') if len(l.split()) > 2 or not l.strip()]
        text = '\n'.join(lines)

        return text.strip()

    def _create_chunk(self, text: str, source: str, chunk_id: int) -> Dict:
        """Crea un chunk con metadata enriquecida."""
        text = text.strip()
        return {
            "text": text,
            "metadata": {
                "source": source,
                "chunk_id": chunk_id,
                "word_count": len(text.split()),
                "section_type": self.detect_section_type(text),
                "clinical_topic": self.extract_clinical_topic(text),
            }
        }

    # ------------------------------------------------------------------
    # PROCESAMIENTO PRINCIPAL
    # ------------------------------------------------------------------

    def process_knowledge_base(self) -> Tuple[List[Dict], np.ndarray]:
        """Procesa todos los PDFs y genera chunks + embeddings."""
        all_chunks: List[Dict] = []
        pdf_files = list(self.kb_dir.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"❌ No se encontraron PDFs en {self.kb_dir}")

        print(f"\n📚 Procesando {len(pdf_files)} documentos PDF...")

        global_chunk_id = 0
        failed_pdfs = []

        for pdf_file in tqdm(pdf_files, desc="Extrayendo texto"):
            text = self.extract_text_from_pdf(pdf_file)

            if not text:
                failed_pdfs.append(pdf_file.name)
                continue

            chunks = self.intelligent_clinical_chunking(
                text,
                source=pdf_file.name,
                start_id=global_chunk_id,
            )
            global_chunk_id += len(chunks)
            all_chunks.extend(chunks)

            status = f"{len(chunks)} chunks"
            if len(chunks) == 0:
                status += " ⚠️  (sin contenido clínico detectado)"
            print(f"   {pdf_file.name}: {status}")

        if failed_pdfs:
            print(f"\n⚠️  PDFs sin texto extraíble ({len(failed_pdfs)}):")
            for f in failed_pdfs:
                print(f"   - {f}")

        print(f"\n✅ Total de chunks generados: {len(all_chunks)}")

        if not all_chunks:
            raise ValueError(
                "❌ No se generaron chunks.\n"
                "   Revisa que los PDFs no sean imágenes puras (requieren OCR)\n"
                "   o prueba reduciendo min_words."
            )

        print(f"\n🧠 Generando embeddings para {len(all_chunks)} chunks...")
        texts = [chunk["text"] for chunk in all_chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return all_chunks, embeddings

    # ------------------------------------------------------------------
    # ÍNDICE FAISS
    # ------------------------------------------------------------------

    def create_faiss_index(
        self,
        embeddings: np.ndarray,
        use_quantization: bool = True,
    ) -> faiss.Index:
        """Crea índice FAISS optimizado para móvil con nprobe proporcional."""
        n_vectors, dim = embeddings.shape
        print(f"\n🔨 Creando índice FAISS ({n_vectors} vectores, dim={dim})...")

        if use_quantization and n_vectors > 1000:
            n_clusters = min(int(np.sqrt(n_vectors)), 100)
            m = 8
            bits = 8

            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, n_clusters, m, bits)

            print(f"   Entrenando IVF-PQ (clusters={n_clusters}, m={m}, bits={bits})...")
            index.train(embeddings)
            index.add(embeddings)

            # nprobe proporcional — busca ~10% de los clusters (mínimo 1)
            index.nprobe = max(1, n_clusters // 10)
            print(f"   nprobe={index.nprobe} de {n_clusters} clusters")

            orig_mb = n_vectors * dim * 4 / (1024 ** 2)
            comp_mb = n_vectors * m * bits / 8 / (1024 ** 2)
            print(f"   📊 Compresión: {orig_mb:.1f}MB → {comp_mb:.1f}MB ({comp_mb/orig_mb*100:.1f}%)")

        else:
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            print(f"   Índice plano (n={n_vectors} < 1000, sin cuantización)")

        return index

    # ------------------------------------------------------------------
    # GUARDADO DE ARTEFACTOS
    # ------------------------------------------------------------------

    def save_artifacts(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        index: faiss.Index,
    ):
        """Guarda todos los artefactos del RAG."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # chunks.json
        chunks_file = self.output_dir / "chunks.json"
        print(f"\n💾 Guardando {chunks_file}...")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # index.faiss
        index_file = self.output_dir / "index.faiss"
        print(f"💾 Guardando {index_file}...")
        faiss.write_index(index, str(index_file))

        # embeddings.npy (debugging / reindexado)
        embeddings_file = self.output_dir / "embeddings.npy"
        print(f"💾 Guardando {embeddings_file}...")
        np.save(embeddings_file, embeddings)

        # embeddings.bin — para RAG híbrido C++
        # Header: [int32 n, int32 dim] + float32 row-major
        bin_file = self.output_dir / "embeddings.bin"
        print(f"💾 Guardando {bin_file} (RAG C++)...")
        n, dim = embeddings.shape
        with open(bin_file, "wb") as bf:
            bf.write(np.int32(n).tobytes())
            bf.write(np.int32(dim).tobytes())
            bf.write(embeddings.astype(np.float32).tobytes(order="C"))

        # metadata.json
        model_name = self._get_model_name()
        metadata = {
            "num_chunks": len(chunks),
            "embedding_dim": int(self.embedding_dim),
            "model_name": model_name,
            "min_words": self.min_words,
            "max_words": self.max_words,
            "overlap_sentences": self.overlap_sentences,
            "index_type": type(index).__name__,
            "total_size_mb": round(self._calculate_total_size(), 2),
            "topics": self._count_topics(chunks),
            "sources": self._count_sources(chunks),
        }

        metadata_file = self.output_dir / "metadata.json"
        print(f"💾 Guardando {metadata_file}...")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Artefactos guardados en: {self.output_dir}")
        self._print_summary(metadata)

    def _get_model_name(self) -> str:
        """Obtiene nombre del modelo con múltiples fallbacks."""
        if hasattr(self.model, 'model_card_data') and self.model.model_card_data:
            name = getattr(self.model.model_card_data, 'model_name', None)
            if name:
                return name
        if hasattr(self.model, '_model_config'):
            name = self.model._model_config.get('model_name_or_path', None)
            if name:
                return name
        if hasattr(self.model, '_model_card_vars'):
            return self.model._model_card_vars.get('model_name', 'unknown')
        return 'unknown'

    def _calculate_total_size(self) -> float:
        return sum(f.stat().st_size for f in self.output_dir.glob("*") if f.is_file()) / (1024 ** 2)

    def _count_topics(self, chunks: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for chunk in chunks:
            t = chunk["metadata"].get("clinical_topic", "Desconocido")
            counts[t] = counts.get(t, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def _count_sources(self, chunks: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for chunk in chunks:
            s = chunk["metadata"].get("source", "Desconocido")
            counts[s] = counts.get(s, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def _print_summary(self, metadata: Dict):
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE GENERACIÓN RAG")
        print("=" * 60)
        print(f"📄 Chunks:         {metadata['num_chunks']}")
        print(f"🧠 Dimensión:      {metadata['embedding_dim']}")
        print(f"📦 Tamaño total:   {metadata['total_size_mb']:.2f} MB")
        print(f"🔧 Índice:         {metadata['index_type']}")
        print(f"📏 Palabras/chunk: {metadata['min_words']} - {metadata['max_words']}")
        print(f"🔁 Overlap:        {metadata['overlap_sentences']} oraciones")
        print(f"🤖 Modelo:         {metadata['model_name']}")

        print("\n📋 Temas clínicos (top 10):")
        for topic, count in list(metadata['topics'].items())[:10]:
            pct = count / metadata['num_chunks'] * 100
            bar = "█" * max(1, int(pct / 3))
            print(f"   {topic:<30} {count:>4} ({pct:4.1f}%) {bar}")

        print("\n📁 Fuentes (top 15):")
        for source, count in list(metadata['sources'].items())[:15]:
            print(f"   {source:<52} {count:>4} chunks")

        print("=" * 60)
        print("\n✅ Base de datos RAG lista para integración en móvil!")

    # ------------------------------------------------------------------
    # PRUEBA DE BÚSQUEDA
    # ------------------------------------------------------------------

    def test_search(self, index: faiss.Index, chunks: List[Dict], k: int = 3):
        """Prueba de búsqueda semántica con queries clínicas de ejemplo."""
        print("\n🔍 PRUEBA DE BÚSQUEDA")
        print("=" * 60)

        test_queries = [
            "¿Cuáles son los criterios diagnósticos de preeclampsia severa?",
            "Manejo del trabajo de parto prematuro con membranas íntegras",
            "Dosis de sulfato de magnesio en eclampsia",
            "Protocolo de hemorragia postparto por atonía uterina",
            "Criterios de ingreso a UCI en paciente obstétrica",
            "Criterios de sepsis materna qSOFA SOFA",
        ]

        for query in test_queries:
            print(f"\n❓ {query}")
            q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
            distances, indices = index.search(q_emb, k)

            for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
                if idx < 0 or idx >= len(chunks):
                    continue
                chunk = chunks[idx]
                preview = chunk["text"][:180].replace("\n", " ") + "..."
                print(f"   {i}. [{dist:.3f}] {chunk['metadata']['source']} | "
                      f"{chunk['metadata']['clinical_topic']} | {chunk['metadata']['word_count']}w")
                print(f"      {preview}\n")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    BASE_DIR = Path(__file__).parent.parent
    KB_DIR = BASE_DIR / "knowledge_base"
    OUTPUT_DIR = BASE_DIR / "embeddings"

    print("=" * 60)
    print("🚀 GENERADOR DE BASE DE DATOS RAG PARA MÓVIL")
    print("   Obstetricia · Modelos 3B · Edge · OCR habilitado")
    print("=" * 60)

    print("\n📦 Dependencias requeridas:")
    print("   pip install sentence-transformers faiss-cpu pdfplumber pymupdf pillow tqdm surya-ocr")
    print("   (surya-ocr descarga modelos ~1GB la primera vez que procesa un PDF escaneado)\n")

    generator = MedicalRAGGenerator(
        knowledge_base_dir=str(KB_DIR),
        output_dir=str(OUTPUT_DIR),
        # MiniLM: ~120MB, rápido, óptimo para edge/móvil
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        min_words=60,
        max_words=120,
        overlap_sentences=2,
        # Si el texto extraído tiene menos de 80 palabras → activar surya OCR
        min_text_words_for_ocr=80,
    )

    chunks, embeddings = generator.process_knowledge_base()
    index = generator.create_faiss_index(embeddings, use_quantization=True)
    generator.save_artifacts(chunks, embeddings, index)
    generator.test_search(index, chunks, k=3)

    print("\n✅ Proceso completado!")
    print(f"📁 Artefactos en: {OUTPUT_DIR}")
    print("\nArchivos generados:")
    print("  chunks.json      → Fragmentos con metadata clínica")
    print("  index.faiss      → Índice vectorial IVF-PQ")
    print("  embeddings.npy   → Embeddings raw (debugging)")
    print("  embeddings.bin   → Embeddings binarios para C++")
    print("  metadata.json    → Info del sistema y distribución")


if __name__ == "__main__":
    main()
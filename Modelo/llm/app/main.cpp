#include "llama.h"
#include "ggml.h"
#include "rag_engine.h"
#include "clinical_engine.h"
#include "text_norm.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <libgen.h>
#endif

namespace {

const char* FINETUNED_MODEL = "models/qwen-medicina-q4km.gguf";

const char* SYSTEM_PROMPT =
    "Eres ObsIA, un medico obstetra experto. "
    "Responde en espanol con claridad, rigor y tono profesional. "
    "Usa exclusivamente el contexto RAG provisto en el mensaje del usuario; no inventes ni completes con conocimientos externos. "
    "Solo puedes agregar preguntas de aclaracion y advertencias de seguridad basadas en los sintomas del usuario. "
    "Si el contexto es insuficiente, indicalo y pide los datos clinicos faltantes. "
    "Evita terminos incompatibles con el estado obstetrico. "
    "Si hay signos de alarma, recomienda atencion inmediata. "
    "Responde en 3 a 5 frases. ";

std::string normalize_lower(const std::string& s) {
    return normalize_spanish_lower(s);
}

std::string truncate_text(const std::string& s, size_t max_len) {
    if (s.size() <= max_len) return s;
    return s.substr(0, max_len) + "...";
}

std::string join_items(const std::vector<std::string>& items, size_t max_items, const char* sep) {
    std::ostringstream oss;
    size_t added = 0;
    for (const auto& it : items) {
        if (it.empty()) continue;
        if (added > 0) oss << sep;
        oss << truncate_text(it, 120);
        added++;
        if (added >= max_items) break;
    }
    return oss.str();
}

std::string pick_urgent_warning(const std::vector<std::string>& warnings) {
    for (const auto& w : warnings) {
        std::string lw = normalize_lower(w);
        if (lw.find("emergencia") != std::string::npos ||
            lw.find("atencion inmediata") != std::string::npos ||
            lw.find("urgente") != std::string::npos ||
            lw.find("sangrado") != std::string::npos ||
            lw.find("convulsion") != std::string::npos ||
            lw.find("severo") != std::string::npos) {
            return w;
        }
    }
    return "";
}

bool validate_response(const std::string& response, const ClinicalPlan& plan, const std::string& rag_context) {
    if (response.size() < 20) return false;

    const std::string out = normalize_lower(response);
    const std::string rag = normalize_lower(rag_context);

    for (const auto& term : plan.banned_terms) {
        if (term.empty()) continue;
        if (out.find(term) != std::string::npos) return false;
    }

    int required_total = 0;
    int required_hits = 0;
    for (const auto& term : plan.required_keywords) {
        if (term.empty()) continue;
        if (rag.find(term) != std::string::npos) {
            required_total++;
            if (out.find(term) != std::string::npos) required_hits++;
        }
    }
    if (required_total > 0 && required_hits == 0) return false;

    return true;
}

std::string build_safe_response(const ClinicalPlan& plan, const std::string& rag_context, bool rag_missing) {
    std::ostringstream oss;
    if (rag_missing) {
        oss << "No se encontro contexto clinico en la base RAG para tu consulta. ";
    } else {
        oss << "No pude generar una respuesta confiable solo con el contexto disponible. ";
    }

    if (!rag_context.empty()) {
        if (!plan.evidence.empty()) {
            oss << "Del contexto se desprende: ";
            const size_t max_ev = std::min<size_t>(2, plan.evidence.size());
            for (size_t i = 0; i < max_ev; ++i) {
                if (i > 0) oss << "; ";
                oss << truncate_text(plan.evidence[i], 160);
            }
            oss << ". ";
        } else {
            oss << "El contexto no aporta detalles clinicos suficientes. ";
        }
    }

    if (!plan.questions.empty()) {
        oss << "Faltan datos: " << join_items(plan.questions, 2, "; ") << ". ";
    }

    std::string urgent = pick_urgent_warning(plan.warnings);
    if (!urgent.empty()) {
        if (!urgent.empty() && urgent.back() != '.') urgent += ".";
        oss << urgent;
    }

    std::string out = oss.str();
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) {
        out.pop_back();
    }
    return out;
}

// Obtener el directorio del ejecutable para resolver rutas relativas
std::string get_exe_dir() {
#ifdef _WIN32
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    std::string s(path);
    size_t pos = s.find_last_of("\\/");
    return (pos != std::string::npos) ? s.substr(0, pos) : ".";
#else
    char path[1024];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len == -1) return ".";
    path[len] = '\0';
    std::string s(path);
    size_t pos = s.find_last_of('/');
    return (pos != std::string::npos) ? s.substr(0, pos) : ".";
#endif
}

// Resuelve una ruta relativa respecto al directorio del proyecto (2 niveles arriba del exe)
std::string resolve_path(const std::string& relative_path, const std::string& exe_dir) {
    // exe esta en: <proyecto>/build/bin/<config>/obgyn_chat.exe  (3 niveles)
    //          o:  <proyecto>/build/bin/obgyn_chat.exe           (2 niveles)
    // Necesitamos llegar a <proyecto>/

    // Verificar primero si el archivo existe con la ruta tal cual (ruta absoluta o CWD)
    {
        FILE* f = fopen(relative_path.c_str(), "r");
        if (f) { fclose(f); return relative_path; }
    }

    // Probar subiendo niveles desde el exe
    std::string base = exe_dir;
    for (int up = 1; up <= 4; ++up) {
        size_t pos = base.find_last_of("\\/");
        if (pos == std::string::npos) break;
        base = base.substr(0, pos);

        std::string candidate = base + "/" + relative_path;
        FILE* f = fopen(candidate.c_str(), "r");
        if (f) { fclose(f); return candidate; }
    }

    // Fallback: devolver la ruta original
    return relative_path;
}

void print_usage(const char* argv0) {
    printf("\nUso: %s [opciones]\n", argv0);
    printf("  -c   TamaÃ±o de contexto (por defecto: 1536)\n");
    printf("  -ngl Capas en GPU, 0=solo CPU (por defecto: 0)\n");
    printf("  -r   Ruta al archivo chunks.json del RAG\n");
    printf("  -k   NÂº de fragmentos RAG a inyectar (por defecto: 3)\n");
    printf("  -t   NÃºmero de hilos CPU (por defecto: 2)\n");
    printf("  -b   Batch size para prefill (por defecto: 128)\n");
    printf("  -ub  Micro-batch size (por defecto: 64)\n");
    printf("\nEjemplo: %s -k 3 -t 4\n\n", argv0);
}

// GestiÃ³n de ventana de contexto (Rolling Context)
void ensure_kv_space(llama_context* ctx, int n_needed, int n_keep_system) {
    const int n_ctx = llama_n_ctx(ctx);
    int n_past = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
    if (n_past < 0) n_past = 0;

    if (n_past + n_needed > n_ctx) {
        int n_discard = (n_past - n_keep_system) / 2;
        if (n_discard < n_needed) n_discard = n_needed;

        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep_system, n_keep_system + n_discard);
        llama_memory_seq_add(llama_get_memory(ctx), 0, n_keep_system + n_discard, n_past, -n_discard);
    }
}

} // namespace

int main(int argc, char** argv) {
    // === PRIMERO: Silenciar logs antes de cualquier operaciÃ³n de llama/ggml ===
    auto log_cb = [](enum ggml_log_level level, const char* text, void* /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            // Filtrar el warning especÃ­fico de ggml_backend_init que no es error real
            if (strstr(text, "ggml_backend_init") != nullptr) return;
            fprintf(stderr, "%s", text);
        }
    };
    ggml_log_set(log_cb, nullptr);
    llama_log_set(log_cb, nullptr);

    std::string exe_dir = get_exe_dir();
    std::string model_path = FINETUNED_MODEL;
    std::string rag_path = "rag/embeddings/chunks.json";
    int ngl = 0;
    int n_ctx = 1536;    // Mas contexto para RAG
    int n_threads = 2;   // Menos hilos = menos CPU
    int n_batch = 128;   // Menor batch = menos RAM pico
    int n_ubatch = 64;   // Micro-batch mas chico
    int rag_k = 3;       // Mas RAG = mas contexto

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0) {
            if (i + 1 < argc) rag_path = argv[++i];
        } else if (strcmp(argv[i], "-k") == 0) {
            if (i + 1 < argc) rag_k = std::max(1, std::min(10, std::stoi(argv[++i])));
        } else if (strcmp(argv[i], "-c") == 0) {
            if (i + 1 < argc) n_ctx = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0) {
            if (i + 1 < argc) ngl = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) n_threads = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch-size") == 0) {
            if (i + 1 < argc) n_batch = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-ub") == 0) {
            if (i + 1 < argc) n_ubatch = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (n_ctx < 256) n_ctx = 256;
    if (n_batch > n_ctx) n_batch = n_ctx;
    if (n_ubatch > n_batch) n_ubatch = n_batch;
    if (n_threads < 1) n_threads = 1;

    // Resolver rutas relativas desde el directorio del ejecutable
    model_path = resolve_path(model_path, exe_dir);
    rag_path = resolve_path(rag_path, exe_dir);

    printf("ðŸ“‚ Directorio exe: %s\n", exe_dir.c_str());
    printf("ðŸ“‚ Modelo finetuneado: %s\n", model_path.c_str());
    printf("ðŸ“‚ RAG: %s\n", rag_path.c_str());

    // Inicializar RAG
    std::unique_ptr<RAGEngine> rag;
    try {
        rag = std::make_unique<RAGEngine>(rag_path);
        printf("âœ… RAG cargado: %s\n", rag_path.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: RAG no disponible: %s\n", e.what());
        return 1;
    }

    ClinicalEngine clinical_engine;

    // No cargar backends dinamicos aqui: el backend CPU ya esta registrado en builds normales

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error: no se pudo cargar el modelo: %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_ubatch;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = std::min(n_threads, 2);

    // Optimizaciones de memoria
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.type_k = GGML_TYPE_Q8_0;
    ctx_params.type_v = GGML_TYPE_Q8_0;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: no se pudo crear el contexto\n");
        llama_model_free(model);
        return 1;
    }

    // === Sampler optimizado anti-repeticiÃ³n ===
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // Ventana de 256 tokens, penalty moderado para evitar bucles sin apagar demasiado el modelo
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, 1.20f, 0.05f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.08f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.35f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const int MAX_GEN_TOKENS = 260;  // Limite duro: respuesta breve

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int system_tokens_len = 0;

    // --- Funciones auxiliares ---

    auto run_inference = [&](const std::vector<llama_token>& tokens, bool is_system = false) -> bool {
        if (is_system) {
            if ((int)tokens.size() > llama_n_ctx(ctx)) {
                fprintf(stderr, "Error: System prompt demasiado largo\n");
                return false;
            }
        } else {
            ensure_kv_space(ctx, (int)tokens.size() + 100, system_tokens_len);
        }

        int n_past = 0;
        int max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        if (max_pos != -1) n_past = max_pos + 1;

        for (int i = 0; i < (int)tokens.size(); i += n_batch) {
            int n_eval = std::min((int)tokens.size() - i, n_batch);

            batch.n_tokens = n_eval;
            for (int k = 0; k < n_eval; ++k) {
                batch.token[k] = tokens[i + k];
                batch.pos[k] = n_past + i + k;
                batch.n_seq_id[k] = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k] = false;
            }

            // Logits solo en el Ãºltimo token absoluto
            if (i + n_eval == (int)tokens.size()) {
                batch.logits[n_eval - 1] = true;
            }

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "Error en decode\n");
                return false;
            }
        }
        return true;
    };

    auto generate_response = [&]() -> std::string {
        std::string response;
        int n_past = 0;
        int max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        if (max_pos != -1) n_past = max_pos + 1;

        llama_token new_token_id;
        int gen_count = 0;
        std::vector<llama_token> recent_tokens;

        while (true) {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, new_token_id)) break;

            gen_count++;
            if (gen_count >= MAX_GEN_TOKENS) break;

            // DetecciÃ³n de repeticiÃ³n: patrÃ³n de 3 tokens x 4 repeticiones
            recent_tokens.push_back(new_token_id);
            if (recent_tokens.size() >= 12) {
                size_t sz = recent_tokens.size();
                bool repeating = true;
                for (int r = 1; r < 4 && repeating; r++) {
                    for (int p = 0; p < 3 && repeating; p++) {
                        if (recent_tokens[sz - 12 + p] != recent_tokens[sz - 12 + r*3 + p]) {
                            repeating = false;
                        }
                    }
                }
                if (repeating) break;
            }

            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) break;
            std::string piece(buf, (size_t)n);
            response += piece;

            // Rotar contexto si se llena
            if (n_past + 1 > llama_n_ctx(ctx)) {
                ensure_kv_space(ctx, 10, system_tokens_len);
                max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
                if (max_pos != -1) n_past = max_pos + 1;
            }

            batch.n_tokens = 1;
            batch.token[0] = new_token_id;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            n_past++;

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "Error en decode (gen)\n");
                break;
            }
        }
        return response;
    };

    // --- Chat template ---
    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        fprintf(stderr, "Este modelo no tiene plantilla de chat.\n");
        return 1;
    }

    std::vector<std::pair<std::string, std::string>> message_contents;
    std::vector<char> formatted((size_t)llama_n_ctx(ctx));
    int prev_len = 0;

    // Ingerir System Prompt
    message_contents.push_back({"system", SYSTEM_PROMPT});
    std::vector<llama_chat_message> msg_ptrs;
    for (const auto& m : message_contents) msg_ptrs.push_back({m.first.c_str(), m.second.c_str()});
    int len_sys = llama_chat_apply_template(tmpl, msg_ptrs.data(), (int)msg_ptrs.size(), false, formatted.data(), (int)formatted.size());
    if (len_sys > (int)formatted.size()) {
        formatted.resize((size_t)len_sys);
        len_sys = llama_chat_apply_template(tmpl, msg_ptrs.data(), (int)msg_ptrs.size(), false, formatted.data(), (int)formatted.size());
    }

    {
        std::string sys_text(formatted.begin(), formatted.begin() + len_sys);
        std::vector<llama_token> sys_tokens(sys_text.size() + 2);
        int n_sys = llama_tokenize(vocab, sys_text.c_str(), sys_text.size(), sys_tokens.data(), sys_tokens.size(), true, true);
        if (n_sys > 0) {
            sys_tokens.resize(n_sys);
            if (run_inference(sys_tokens, true)) {
                system_tokens_len = n_sys;
            } else {
                return 1;
            }
        }
    }
    prev_len = len_sys;

    printf("\n  ðŸ©º ObsIA - Obstetra (%d hilos CPU). Escribe tu pregunta.\n", n_threads);
    printf("  Escribe 'salir' para terminar.\n\n");

    while (true) {
        printf("\033[32m> \033[0m");
        std::string user;
        std::getline(std::cin, user);

        if (user.empty() || user == "salir" || user == "exit" || user == "quit") break;

        ObstetricState state = clinical_engine.detect_state(user);

        // Construir mensaje solo con RAG, filtrado por estado obstetrico
        std::string rag_context;
        if (rag) {
            rag_context = rag->get_augmented_context(user, rag_k, state);
        }

        ClinicalPlan plan = clinical_engine.build_plan(user, rag_context);

        if (rag_context.empty()) {
            std::string fallback = build_safe_response(plan, rag_context, true);
            printf("\033[33mObsIA: %s\n\033[0m\n", fallback.c_str());
            continue;
        }

        std::string final_user_msg =
            "Contexto RAG (usa solo esto):\n" + rag_context +
            "\nInstrucciones:\n"
            "- Responde en 3 a 5 frases.\n"
            "- Usa solo informacion del Contexto RAG.\n"
            "- Si falta informacion, indicalo y haz preguntas concretas.\n"
            "- Si hay signos de alarma descritos por el usuario, recomienda atencion inmediata.\n";

        if (!plan.banned_terms.empty()) {
            final_user_msg += "- No menciones: " + join_items(plan.banned_terms, 6, ", ") + ".\n";
        }

        final_user_msg += "Consulta del usuario:\n" + user;

        message_contents.push_back({"user", final_user_msg});
        msg_ptrs.clear();
        for (const auto& m : message_contents) msg_ptrs.push_back({m.first.c_str(), m.second.c_str()});

        int new_len = llama_chat_apply_template(tmpl, msg_ptrs.data(), (int)msg_ptrs.size(), true, formatted.data(), (int)formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize((size_t)new_len * 2);
            new_len = llama_chat_apply_template(tmpl, msg_ptrs.data(), (int)msg_ptrs.size(), true, formatted.data(), (int)formatted.size());
        }

        if (new_len < 0) {
            fprintf(stderr, "Error apply template\n");
            message_contents.pop_back();
            break;
        }

        std::string diff_text(formatted.begin() + prev_len, formatted.begin() + new_len);

        // Ingerir prompt usuario
        {
            std::vector<llama_token> tokens(diff_text.size() + 2);
            int n = llama_tokenize(vocab, diff_text.c_str(), diff_text.size(), tokens.data(), tokens.size(), false, true);
            if (n <= 0) {
                fprintf(stderr, "Error: no se obtuvieron tokens del prompt\n");
                message_contents.pop_back();
                continue;
            }
            tokens.resize(n);
            if (!run_inference(tokens, false)) {
                fprintf(stderr, "Error ingiriendo prompt usuario\n");
                message_contents.pop_back();
                continue;
            }
        }

        std::string model_reply = generate_response();
        // Recortar espacios iniciales si el tokenizer los agrega
        size_t start = 0;
        while (start < model_reply.size() && std::isspace(static_cast<unsigned char>(model_reply[start]))) {
            start++;
        }
        if (start > 0) model_reply = model_reply.substr(start);

        std::string final_reply = model_reply;
        if (!validate_response(model_reply, plan, rag_context)) {
            final_reply = build_safe_response(plan, rag_context, false);
        }

        printf("\033[33mObsIA: %s\n\033[0m\n", final_reply.c_str());

        message_contents.pop_back();  // Quitar el mensaje del usuario
        prev_len = len_sys;

        // Limpiar el contexto para que cada respuesta dependa solo del RAG actual
        int max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        if (max_pos + 1 > system_tokens_len) {
            llama_memory_seq_rm(llama_get_memory(ctx), 0, system_tokens_len, max_pos + 1);
        }
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}







// The 8-lane concurrent-prefill paging proof (NINFER_PAGED8_ARTIFACT names the Inkling 2
// artifact; without it the test skips): eight requests enter at one boundary (submit_group) and
// prefill and decode together under --kv-paging, for three rounds — fresh lanes, the same
// prompts again (the lanes resume their own retained prefixes) and shifted prompts (the lanes
// are trimmed and refilled). Every greedy output must equal a dense control engine's byte for
// byte: the pool relocates bytes, never changes them.

#include "ninfer/engine.h"
#include "ninfer/kv_cache_storage.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "OK   " : "FAIL ") << what << '\n';
    failures += ok ? 0 : 1;
}

ninfer::PromptInput prompt(std::string text) {
    ninfer::PromptInput input;
    input.messages.push_back(ninfer::ChatMessage{
        .role  = ninfer::ChatRole::User,
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = std::move(text)}}});
    input.options.enable_thinking = false;
    return input;
}

std::string list_prompt(int items) {
    const char* words[] = {"alpha",  "beta",   "gamma",   "delta",  "epsilon",
                           "zeta",   "eta",    "theta",   "iota",   "kappa"};
    std::string out = "Remember this list of facts: ";
    for (int j = 0; j < items; ++j) {
        out += std::string(words[j % 10]) + "-" + std::to_string(j) + " is item " +
               std::to_string(j) + ". ";
    }
    out += "\nReply with the single word OK when you have read them.";
    return out;
}

ninfer::RequestOptions request(std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    return options;
}

ninfer::EngineOptions engine_options(const char* artifact, bool paged) {
    using KvCacheStorage = ninfer::KvCacheStorage;
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 32768;
    options.max_concurrency      = 8;
    options.kv_cache             = KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.kv_paging            = paged;
    options.pending_timeout_ms   = 600000;
    options.use_cuda_graph       = true;
    if (const char* context = std::getenv("NINFER_PAGED8_CONTEXT"); context && *context) {
        options.max_context = static_cast<std::uint32_t>(std::atoi(context));
    }
    return options;
}

// Three rounds of eight lanes: fresh, resumed (the same prompts) and shifted (new prompts, the
// lanes trimmed and refilled).
std::vector<std::vector<std::string>> run_engine(ninfer::Engine& engine) {
    std::vector<std::vector<std::string>> texts(3, std::vector<std::string>(8));
    for (int round = 0; round < 3; ++round) {
        const int shift = round == 2 ? 40 : 0;
        std::vector<ninfer::GenerationSubmission> submissions;
        std::vector<ninfer::PreparedPrompt> prompts;
        prompts.reserve(8);
        for (int lane = 0; lane < 8; ++lane) {
            prompts.push_back(engine.prepare(prompt(list_prompt(200 + lane * 5 + shift))));
            ninfer::GenerationSubmission submission;
            submission.prompt           = std::move(prompts.back());
            submission.options          = request(256);
            submission.consumer_mode    = ninfer::OutputConsumerMode::Aggregate;
            submission.pending_deadline = std::chrono::steady_clock::time_point::max();
            submissions.push_back(std::move(submission));
        }
        auto handles = engine.submit_group(std::move(submissions));
        check(handles.size() == 8, "round " + std::to_string(round + 1) + ": eight handles");
        for (int lane = 0; lane < 8; ++lane) {
            const ninfer::GenerationResult result = handles[std::size_t(lane)].wait();
            texts[std::size_t(round)][std::size_t(lane)] = result.content;
        }
    }
    
    return texts;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_PAGED8_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP  NINFER_PAGED8_ARTIFACT is not set\n";
        return 77;
    }
    try {
        std::cout << "dense control engine\n";
        {
            ninfer::Engine dense(engine_options(artifact, false));
            const auto texts = run_engine(dense);
            // Keep the control's outputs for the paging engine in a file-side by side: both
            // engines live in one process? No — the dense engine is destroyed first, the paging
            // one starts on a clean device; the comparison happens below over the saved texts.
            std::FILE* out = std::fopen("paged8-dense-expected.txt", "w");
            for (int round = 0; round < 3; ++round) {
                for (int lane = 0; lane < 8; ++lane) {
                    std::fputs(texts[std::size_t(round)][std::size_t(lane)].c_str(), out);
                    std::fputc('\n', out);
                }
            }
            std::fclose(out);
        }
        std::cout << "paging engine\n";
        std::vector<std::vector<std::string>> paged;
        {
            ninfer::Engine paged_engine(engine_options(artifact, true));
            paged = run_engine(paged_engine);
        }
        std::FILE* in = std::fopen("paged8-dense-expected.txt", "r");
        check(in != nullptr, "the dense control's outputs were saved");
        if (in != nullptr) {
            char buffer[4096];
            for (int round = 0; round < 3; ++round) {
                for (int lane = 0; lane < 8; ++lane) {
                    std::string expected = std::fgets(buffer, sizeof(buffer), in) != nullptr
                                               ? std::string(buffer)
                                               : std::string();
                    if (!expected.empty() && expected.back() == '\n') { expected.pop_back(); }
                    const std::string& got = paged[std::size_t(round)][std::size_t(lane)];
                    check(got == expected, "round " + std::to_string(round + 1) + " lane " +
                                               std::to_string(lane) + ": paged == dense (" +
                                               std::to_string(got.size()) + " vs " +
                                               std::to_string(expected.size()) + " bytes)");
                }
            }
            std::fclose(in);
        }
    } catch (const std::exception& e) {
        std::cout << "FAIL  " << e.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "PASS  paged 8-lane\n" : "FAIL  paged 8-lane\n");
    return failures == 0 ? 0 : 1;
}

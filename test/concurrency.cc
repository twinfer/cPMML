
/*******************************************************************************
 * Concurrent-scoring regression test.
 *
 * R3 in the review was that scoring was not thread-safe despite the
 * thread-local indexer design: the per-model string map was mutated during
 * scoring, and OMP worker threads scored against whichever indexer happened to
 * be active on their thread. Both are fixed -- the map is frozen when parsing
 * ends and every entry point establishes a ScopedIndexer -- but nothing in the
 * suite would notice if either regressed, because a data race does not fail a
 * single-threaded fixture.
 *
 * The assertion is not "it did not crash", which a race passes almost every
 * time. It is that every concurrent answer is identical to the single-threaded
 * one: an interning race shows up as a categorical value getting a different
 * code, which changes the prediction rather than the process's health.
 *
 * Run it under `cmake -DSANITIZE=thread`. That is where it earns its keep:
 * reintroduce the R3 defect -- delete the frozen check in
 * Indexer::get_string_value -- and a plain Release build passes this test every
 * time, while the same binary built with -fsanitize=thread reports the race and
 * fails the divergence assertion as well.
 *******************************************************************************/

#include <atomic>
#include <typeinfo>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "cPMML.h"
#include "utils/csvreader.h"

static std::atomic<int> failures{0};

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << #cond      \
                << "\n";                                                        \
      ++failures;                                                               \
    }                                                                           \
  } while (0)

static const int THREADS = 8;
static const int ROUNDS = 4;

static std::vector<cpmml::Input> read_inputs(const std::string& csv_path) {
  std::vector<cpmml::Input> inputs;
  CSVReader reader(csv_path);

  while (true) {
    const auto row = reader.read();
    if (row.empty()) break;

    cpmml::Input input;
    for (const auto& [field, value] : row) input[field] = value;
    inputs.push_back(std::move(input));
  }

  return inputs;
}

/**
 * A categorical value the DataDictionary never saw, tagged so that no two
 * callers ever present the same string. That distinctness is the whole point:
 * this is the case that used to mutate the shared string map mid-scoring, and
 * if every thread asked for a string some earlier thread had already interned,
 * the concurrent path would only ever read and the race would not arise.
 *
 * The contract being asserted is that an unseen category is not any known
 * category, whatever it spells -- so scoring with tag "t3" must give exactly
 * what scoring with tag "baseline" gave.
 */
static std::vector<cpmml::Input> with_unseen_categories(const std::vector<cpmml::Input>& inputs,
                                                        const std::string& field, const std::string& tag) {
  std::vector<cpmml::Input> result = inputs;
  for (size_t i = 0; i < result.size(); i++)
    if (result[i].count(field)) result[i][field] = "unseen_" + tag + "_" + std::to_string(i);

  return result;
}

static std::vector<std::string> score_all(const cpmml::Model& model, const std::vector<cpmml::Input>& inputs) {
  std::vector<std::string> results;
  results.reserve(inputs.size());
  for (const auto& input : inputs) {
    try {
      results.push_back(model.evaluate(input).as_string());
    } catch (const cpmml::Exception& exception) {
      // A row that cannot be scored must fail the same way on every thread, so
      // the failure is part of the answer being compared -- but by type, not by
      // message: the message embeds the offending sample, which by construction
      // differs between one caller's unseen categories and another's.
      results.push_back(std::string("!") + typeid(exception).name());
    }
  }

  return results;
}

static void concurrent_scoring_matches_serial(const std::string& model_path, const std::string& csv_path,
                                              const std::string& categorical_field) {
  cpmml::Model model(model_path, true);

  const std::vector<cpmml::Input> known = read_inputs(csv_path);
  CHECK(!known.empty());

  // Two baselines, both single-threaded: the rows as given, and the same rows
  // with categorical values nothing has ever seen.
  const std::vector<std::string> expected_known = score_all(model, known);
  const std::vector<std::string> expected_unseen =
      score_all(model, with_unseen_categories(known, categorical_field, "baseline"));

  std::vector<std::thread> workers;
  for (int t = 0; t < THREADS; t++)
    workers.emplace_back([&, t] {
      for (int round = 0; round < ROUNDS; round++) {
        if (score_all(model, known) != expected_known) {
          std::cerr << "FAIL concurrent scoring diverged from serial scoring\n";
          ++failures;
          return;
        }

        // Each thread, each round, presents strings no one has interned.
        const std::string tag = "t" + std::to_string(t) + "r" + std::to_string(round);
        if (score_all(model, with_unseen_categories(known, categorical_field, tag)) != expected_unseen) {
          std::cerr << "FAIL an unseen category scored differently depending on how it was spelled\n";
          ++failures;
          return;
        }
      }
    });

  for (auto& worker : workers) worker.join();
}

/**
 * Parsing sets the thread-local active indexer, so two models being built at
 * once on different threads must not see each other's. The guard in
 * ModelBuilder restores the previous value on every exit path; without it the
 * second model's fields could be interned into the first model's map.
 */
static void concurrent_loading_is_isolated(const std::string& model_path, const std::string& csv_path) {
  const std::vector<cpmml::Input> inputs = read_inputs(csv_path);
  const std::vector<std::string> expected = score_all(cpmml::Model(model_path, true), inputs);

  std::vector<std::thread> workers;
  for (int t = 0; t < THREADS; t++)
    workers.emplace_back([&model_path, &inputs, &expected] {
      for (int round = 0; round < ROUNDS; round++) {
        cpmml::Model model(model_path, true);
        if (score_all(model, inputs) != expected) {
          std::cerr << "FAIL a model built concurrently scored differently\n";
          ++failures;
          return;
        }
      }
    });

  for (auto& worker : workers) worker.join();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: concurrency.exe <model.zip> <dataset.csv> [categorical-field]\n";
    return 2;
  }

  const std::string categorical_field = argc > 3 ? argv[3] : "";
  // Under a sanitizer the two phases have to be separable: a report naming a
  // heap block is useless if a model built on the main thread in one phase can
  // have had its address reused by a model built on a worker in the other.
  const std::string phase = argc > 4 ? argv[4] : "all";

  if (phase == "all" || phase == "score") concurrent_scoring_matches_serial(argv[1], argv[2], categorical_field);
  if (phase == "all" || phase == "load") concurrent_loading_is_isolated(argv[1], argv[2]);

  if (failures > 0) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "concurrency: all checks passed\n";
  return 0;
}

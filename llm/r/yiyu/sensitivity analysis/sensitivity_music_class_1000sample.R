### Sensitivity_Analysis - music genre classification
### Date: 06 Oct 2025
### new parameter grid - varied temp., top_p, top_k (0.1 step)
### fixed - batch_size=10, max_tokens=200, repetition_penalty=1
### Datasets: Music (1000 samples) -> 200 samples for each class
### Sys.setenv(TOGETHER_API_KEY="tgp_v1_uCqf5PWHYRVSzlzTJGfAhnCelb0lQbDfjq5m0Edzyj0")
### setwd("~/Desktop/Integrate project/data analysis/text_analysis_LLM")

## Setup
library(tidyverse)
library(httr)
library(jsonlite)
library(glue)
library(stringr)
library(readr)
library(dplyr)

## Directories
base_dir <- "Sensitivity_Analysis_new_prams"
log_dir <- file.path(base_dir, "logs", "logs_music_genre_1000samples")
list_dir <- file.path(base_dir, "results_list", "music_class_listing_1000samples")
result_dir <- file.path(base_dir, "results", "music_class_1000samples")

dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(list_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

# load data 
set.seed(123)
reduced_classes <- c("rock", "pop/r&b", "electronic", "jazz", "rap")
music_data <- read_csv("data/pitchfork_reviews.csv") %>%
  filter(genre %in% reduced_classes)

# test data - 1000 sample
test_data <- music_data %>%
  group_by(genre) %>%
  slice_sample(n = 200) %>%  # 200 per class = 1000 total
  ungroup() %>%
  transmute(
    id = reviewid,
    review = as.character(review),
    genre = factor(genre, levels = reduced_classes)
  )

print(table(test_data$genre))

# few-shot examples
fewshot_pool <- music_data %>%
  filter(!reviewid %in% test_data$id)

set.seed(42)
few_shot_examples <- list()
for (g in reduced_classes) {
  examples <- fewshot_pool%>% 
    filter(genre == g) %>%
    sample_n(min(3, n()))
  few_shot_examples[[g]] <- examples
}
sapply(few_shot_examples, nrow)

## Define a prompt
## build with real examples
example_texts <- c()
for (genre_class in names(few_shot_examples)) {
  for (i in 1:nrow(few_shot_examples[[genre_class]])) {
    review_text<- few_shot_examples[[genre_class]]$review[i]
    example_texts <- c(example_texts, sprintf("Review: '%s'\nGenre: %s", review_text, genre_class))
  }
}
examples_block <- paste(example_texts, collapse = "\n\n")

music_genre_prompt <- function(batch_size, examples_block, reduced_classes) {
  glue(
    "You are an expert AI music reviewer, designed to classify each music album review into ONE AND ONLY ONE of the following 5 genres:\n",
    "[{paste(reduced_classes, collapse=',')}].\n\n",
    "Below are some example reviews and their corresponding genres:\n",
    "{examples_block}\n\n",
    "I will provide you with a batch of {batch_size} music reviews. Read each review closely and identify any words, phrases, or sentiments that might indicate the genre of the album. Then, assign ONE of 5 genres listed above to each review.\n",
  )
}

build_messages_music_class <- function(batch_reviews, examples_block, reduced_classes) {
  n <- nrow(batch_reviews)
  sys_msg <- glue(
    "You are an expert AI that classifies music reviews into genres.\n",
    "RULES:\n",
    "- Only use these 5 genres: [{paste(reduced_classes, collapse=',')}]. If your answer is not exactly one of these, review and correct before returning it.\n",
    "- NEVER invent new genres, combine genres, or use subgenres or variations.\n",
    "- If multiple fit, choose the most dominant one; if unsure, choose the closest.\n",
    "- Output MUST be a valid JSON array of exactly {n} genres, one per review, in a structured format.\n",
    "- No reasoning, no explanations, no labels, no extra text.\n",
    "- Example output: [\"rap\", \"rock\", \"jazz\", \"electronic\", ...].\n",
    "- Your output must contain exactly {n} genres.",
    "- If there are fewer or more than {n} genres, review and correct the response before returning it.",
    "- Make sure the JSON array is correctly formatted before returning it."
  )
  user_msg <- paste0(
    music_genre_prompt(n, examples_block, reduced_classes), "\n\n",
    "Classify the following reviews:\n",
    jsonlite::toJSON(
      lapply(seq_len(n), function(i) list(id = batch_reviews$id[i], text = batch_reviews$review[i])),
      auto_unbox = TRUE, pretty = TRUE
    )
  )
  
  list(
    list(role = "system", content = sys_msg),
    list(role = "user", content = user_msg)
  )
}

## Split batches
split_batches <- function(data, batch_size) {
  split(data, ceiling(seq_len(nrow(data)) / batch_size))
}

## Output Parsing Helpers
safe_json_parse <- function(txt) {
  tryCatch(jsonlite::fromJSON(txt), error = function(e) NULL)
}

## API config
url <- "https://api.together.xyz/v1/chat/completions"
api_key <- Sys.getenv("TOGETHER_API_KEY")
if (!nzchar(api_key)) {
  stop("API key not found. Please set with Sys.setenv(TOGETHER_API_KEY='your_real_key')")
}

# Batch runner
# if log_all = T, all status will be recorded.
run_batch_music_genre <- function(batch_reviews, param_row, batch_index, log_file,
                                  examples_block, reduced_classes, log_all = FALSE) {
  # build messages (system + user)
  messages <- build_messages_music_class(batch_reviews, examples_block, reduced_classes)
  n <- nrow(batch_reviews)
  
  body_list <- list(
    model = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages = messages,
    temperature = param_row$temperature,
    top_p = param_row$top_p,
    top_k = param_row$top_k,
    repetition_penalty = 1,
    max_tokens = param_row$max_tokens,
    stream = FALSE
  )
  
  t0 <- Sys.time()
  response <- httr::POST(
    url,
    httr::add_headers(Authorization = paste("Bearer", api_key),
                      `Content-Type` = "application/json"),
    body = body_list,
    encode = "json"
  )
  t1 <- Sys.time()
  latency_sec <- as.numeric(difftime(t1, t0, units = "secs"))
  status <- tryCatch(httr::status_code(response), error = function(e) NA_integer_)
  
  result <- try(httr::content(response, as = "parsed", type = "application/json"), silent = TRUE)
  
  assistant_reply <- NA_character_
  if (!inherits(result, "try-error") && !is.null(result$choices)) {
    assistant_reply <- paste(result$choices[[1]]$message$content, collapse = " ")
    assistant_reply <- str_trim(assistant_reply)
  }
  
  # parse classification output
  parsed_labels <- rep(NA_character_, n)
  if (!is.na(assistant_reply)) {
    out <- safe_json_parse(assistant_reply)
    if (!is.null(out) && length(out) == n) {
      parsed_labels <- tolower(trimws(as.character(out)))
      parsed_labels[!parsed_labels %in% tolower(reduced_classes)] <- NA_character_
    }
  }
  
  # logging
  if (!is.null(log_file)) {
    na_rate <- mean(is.na(parsed_labels))
    if (log_all || status != 200 || na_rate > 0) {
      # if (isTRUE(status != 200) || na_rate > 0) { 
      head_txt <- substr(as.character(assistant_reply), 1, 300)
      msg <- paste0(
        "[batch ", batch_index, "] status=", status,
        " | latency=", round(latency_sec, 2), "s",
        " | NA_rate=", round(na_rate, 3),
        " | params={temp=", param_row$temperature,
        ", top_p=", param_row$top_p,
        ", top_k=", param_row$top_k,
        ", max_tokens=", param_row$max_tokens,
        ", batch_size=", param_row$batch_size, "}",
        " | reply_head=", head_txt
      )
      write(paste0(msg, "\n"), file = log_file, append = TRUE)
    }
  }
  
  tibble(
    dataset = "music_genre",
    id = batch_reviews$id,
    Review = batch_reviews$review,
    GroundTruth = batch_reviews$genre,
    Prediction = parsed_labels,
    batch_index = batch_index,
    latency_sec = latency_sec,
    temperature = param_row$temperature,
    top_p = param_row$top_p,
    top_k = param_row$top_k,
    repetition_penalty = 1,
    max_tokens = param_row$max_tokens,
    batch_size = n
  )
}

# experiment runner for music genre classification
run_exp_chunk_music_genre <- function(dataset, param_chunk, chunk_id = 1,
                                      start_batch = 1, log_dir,
                                      examples_block, reduced_classes) {
  batch_counter <- start_batch
  
  for (i in 1:nrow(param_chunk)) {
    param_row <- param_chunk[i, ]
    param_id <- paste0("chunk", chunk_id, "_param", i)
    
    LOG_FILE <- file.path(
      log_dir,
      glue("music_genre_newgrid_{param_id}_{format(Sys.time(), '%Y-%m-%d-%H%M%S')}.log")
    )
    output_file <- file.path(list_dir, paste0(param_id, ".csv"))
    
    if (file.exists(output_file)) {
      cat(glue("[{Sys.time()}] Skipping {param_id} (already exists)\n"))
      next
    }
    cat(glue("[{Sys.time()}] Running {param_id} with parameters: ",
             "T={param_row$temperature}, P={param_row$top_p}, K={param_row$top_k}, ",
             "max_tokens={param_row$max_tokens}, batch_size={param_row$batch_size}\n"))

    batches <- split_batches(dataset, param_row$batch_size)
    combo_results <- vector("list", length(batches))
    
    for (b in seq_along(batches)) {
      res_b <- run_batch_music_genre(batches[[b]], param_row, batch_counter, 
                                     LOG_FILE, examples_block, reduced_classes)
      combo_results[[b]] <- res_b
      batch_counter <- batch_counter + 1
    }
      
    result_df <- bind_rows(combo_results)
    write_csv(result_df, output_file)
    cat(glue("[{Sys.time()}] Finished {param_id}, saved {nrow(result_df)} rows\n"))
    Sys.sleep(0.2)
  }
  return(batch_counter)
}

new_grid <- expand.grid(
  temperature = round(seq(0.1, 1.0, by = 0.1), 1),   # step = 0.1
  top_p       = round(seq(0.1, 1.0, by = 0.1), 1),   # step = 0.1
  top_k       = seq(10, 100, by = 10),               # step = 10
  max_tokens  = c(200),                              # fixed
  batch_size  = c(10),                               # fixed
  stringsAsFactors = FALSE
)
nrow(new_grid) # 1000

todo_grid <- head(new_grid, 10)

# split grid into chunks
param_chunks <- split(todo_grid, ceiling(seq_len(nrow(todo_grid)) / 10))

# run experiment
global_batch <- 1
for (i in seq_along(param_chunks)) {
  cat(glue("\n[{Sys.time()}] === Running chunk {i}/{length(param_chunks)} ===\n"))
  global_batch <- run_exp_chunk_music_genre(
    test_data, 
    param_chunks[[i]], 
    chunk_id = i, 
    start_batch = global_batch, 
    log_dir = log_dir,
    examples_block = examples_block,
    reduced_classes = reduced_classes)
}

# merge all result CSVs into one final file
all_new_csvs <- list.files(list_dir, pattern = "\\.csv$", full.names = TRUE)
all_new_results <- purrr::map_dfr(all_new_csvs, read_csv, show_col_types = FALSE)

final_file <- file.path(result_dir, paste0(
  "music_class_sensitivity_newgrid_all_results_1000sample_", 
  format(Sys.time(), "%Y-%m-%d-%H%M%S"), ".csv"))

write_csv(all_new_results, final_file)





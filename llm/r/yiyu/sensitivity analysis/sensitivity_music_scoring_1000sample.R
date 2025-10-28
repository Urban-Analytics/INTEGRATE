### Sensitivity_Analysis - music scoring
### Date: 06 Oct 2025
### new parameter grid - varied temp., top_p, top_k (0.1 step)
### fixed - batch_size=10, max_tokens=200, repetition_penalty=1
### Datasets: 1-100 scoring - Music (1000 samples)
### Sys.setenv(TOGETHER_API_KEY="tgp_v1_uCqf5PWHYRVSzlzTJGfAhnCelb0lQbDfjq5m0Edzyj0")
### setwd("~/Desktop/Integrate project/data analysis/text_analysis_LLM")

# Setup
library(tidyverse)
library(httr)
library(jsonlite)
library(glue)
library(dplyr)
library(purrr)
library(readr)
library(stringr)
library(ggrepel)

# load data (subset 1000 reviews)
set.seed(123) 
music_data <- read_csv("data/pitchfork_reviews.csv") %>%
  transmute(id = reviewid,
            review = as.character(review),
            score = as.numeric(score) * 10  # scale to 0–100
  ) %>%
  filter(!is.na(review), !is.na(score)) %>%
  slice_sample(n = 1000)   # select 1000 reviews

# create a new results directory to avoid overwriting old results.
base_dir <- "Sensitivity_Analysis_new_prams"
log_dir <- file.path(base_dir, "logs", "logs_music_scoring", "logs_1000sample")
list_dir <- file.path(base_dir, "results_list", "music_scoring_listing_1000sample")
result_dir <- file.path(base_dir, "results", "music_scoring_1000sample")

dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(list_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

# batch split
split_batches <- function(data, batch_size) {
  split(data, ceiling(seq_len(nrow(data)) / batch_size))
}

# prompt design (few-shot examples)
generate_prompt_music <- function(batch_size) {
  glue(
    "You are an expert AI music reviewer, designed to assess and score the quality of music albums based on written reviews. 
     Pitchfork reviews typically score most albums between 60 and 80, with truly exceptional albums sometimes scoring above 90, and poor albums below 50.
     Please use the full range where appropriate, but avoid clustering all scores near the mean. 
     Do not assign most reviews similar scores - use lower and higher scores when justified by the review content.

     Here are some example reviews and their corresponding scores:
     1. ‘A breathtaking, genre-defying album that rewrites the rules of pop music.’ → 96
     2. ‘Though the production is clean, most tracks are uninspired and forgettable.’ → 52
     3. ‘A solid indie rock record with a few standout songs but too much filler.’ → 68
     4. ‘Truly disappointing, lacking any memorable hooks or emotion.’ → 37
     5. ‘Inventive, emotionally resonant, and brilliantly produced.’ → 88

     I will provide you with a batch of {batch_size} music reviews. 
     Your task is to analyze the text in each review and assign a numerical score on a 0-100 range, 
     Scoring rules:
      - 0–20: very poor
      - 21–40: below average
      - 41–60: average
      - 61–80: good to very good
      - 81–90: excellent
      - 91–100: exceptional"
  )
}

build_messages_music <- function(batch_reviews) {
  n <- nrow(batch_reviews)
  sys_msg <- glue(
    "You are an AI that outputs scores for music reviews.
     Output ONLY a valid JSON array of exactly {n} integers (0-100), one per review, in a structured format.
     No reasoning, no explanations, no labels, no extra text.
     Example: [87, 74, 92, 51, 69, ...].
     Ensure the JSON array length matches {n} before returning it."
  )
  user_msg <- paste0(
    generate_prompt_music(n), "\n\n",
    paste0(seq_along(batch_reviews$review), ". ", batch_reviews$review, collapse = "\n")
  )
  list(
    list(role = "system", content = sys_msg),
    list(role = "user", content = user_msg)
  )
}

# parse LLM output
extract_numeric_vec <- function(x) {
  if (is.numeric(x)) return(as.numeric(x))
  if (is.list(x)) return(as.numeric(unlist(lapply(x, extract_numeric_vec))))
  return(numeric(0))
}

clamp01_100_int <- function(v) {
  v <- round(as.numeric(v))
  v[!is.finite(v)] <- NA_real_
  list(
    raw = v,
    clamped = pmin(pmax(v, 0), 100),
    out_of_range = ifelse(is.na(v), NA_integer_, as.integer(v < 0 | v > 100))
  )
}

parse_scores_continuous <- function(raw_output, n_expected) {
  txt <- trimws(if (is.null(raw_output)) "" else raw_output)
  if (startsWith(txt, "[") && !grepl("\\]$", txt) && nchar(txt) < 20000) {
    txt <- paste0(txt, "]")
  }
  parsed <- try(jsonlite::fromJSON(txt), silent = TRUE)
  nums <- NULL
  if (!inherits(parsed, "try-error")) nums <- extract_numeric_vec(parsed)
  
  if ((is.null(nums) || length(nums) == 0) && nzchar(txt)) {
    m <- regexpr("\\[[^\\]]*\\]", txt, perl = TRUE)
    if (m[1] != -1) {
      candidate <- substr(txt, m[1], m[1] + attr(m, "match.length") - 1)
      parsed2 <- try(jsonlite::fromJSON(candidate), silent = TRUE)
      if (!inherits(parsed2, "try-error")) nums <- extract_numeric_vec(parsed2)
    }
  }
  if (is.null(nums) || length(nums) == 0) {
    matches <- gregexpr("[-+]?[0-9]*\\.?[0-9]+", txt, perl = TRUE)
    if (matches[[1]][1] != -1) nums <- as.numeric(regmatches(txt, matches)[[1]])
  }
  if (is.null(nums) || length(nums) == 0) {
    return(list(pred = rep(NA_real_, n_expected), out_of_range = rep(NA_integer_, n_expected)))
  }
  if (length(nums) < n_expected) nums <- c(nums, rep(NA_real_, n_expected - length(nums)))
  if (length(nums) > n_expected) nums <- nums[seq_len(n_expected)]
  adj <- clamp01_100_int(nums)
  list(pred = adj$clamped, out_of_range = adj$out_of_range)
}

# API config
url <- "https://api.together.xyz/v1/chat/completions"
api_key <- Sys.getenv("TOGETHER_API_KEY")
if (!nzchar(api_key)) {
  stop("API key not found. Please set with Sys.setenv(TOGETHER_API_KEY='your_real_key')")
}

# batch runner
run_batch_music <- function(batch_reviews, param_row, batch_index, log_file) {
  messages <- build_messages_music(batch_reviews)
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
    httr::add_headers(Authorization = paste("Bearer", api_key), `Content-Type` = "application/json"),
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
  
  parsed <- list(pred = rep(NA_real_, n), out_of_range = rep(NA_integer_, n))
  if (!is.na(assistant_reply)) parsed <- parse_scores_continuous(assistant_reply, n)
  
  # logging (record NA predictions)
  if (!is.null(log_file)) {
    na_rate <- mean(is.na(parsed$pred))
    if (isTRUE(status != 200) || na_rate > 0) { 
      # head_txt <- substr(as.character(assistant_reply), 1, 300)
      msg <- paste0(
        "[batch ", batch_index, "] status=", status,
        " | latency=", round(latency_sec, 2), "s",
        " | NA_rate=", round(na_rate, 3),
        " | params={temp=", param_row$temperature,
        ", top_p=", param_row$top_p,
        ", top_k=", param_row$top_k,
        ", max_tokens=", param_row$max_tokens,
        ", batch_size=", param_row$batch_size, "}"
        # " | reply_head=", head_txt
      )
      write(paste0(msg, "\n"), file = log_file, append = TRUE)
    }
  }
  
  tibble(
    dataset = "music",
    id = batch_reviews$id,
    Review = batch_reviews$review,
    GroundTruth = batch_reviews$score,
    Prediction = parsed$pred,
    OutOfRangeFlag = parsed$out_of_range,
    batch_index = batch_index,
    latency_sec = latency_sec,
    temperature = param_row$temperature,
    top_p = param_row$top_p,
    top_k = param_row$top_k,
    max_tokens = param_row$max_tokens,
    batch_size = nrow(batch_reviews)
    # batch_size = n
  )
}

# experiment runner
run_exp_chunk_music <- function(dataset, param_chunk, chunk_id = 1, start_batch = 1, log_dir) {
  # batch_counter <- 1
  batch_counter <- start_batch
  
  # logging (per chunk)
  LOG_FILE <- file.path(
    log_dir, 
    paste0("music_scoring_newgrid", chunk_id, "_", format(Sys.time(), "%Y-%m-%d-%H%M%S"), ".log")
  )
  
  for (i in 1:nrow(param_chunk)) {
    param_row <- param_chunk[i, ]
    param_id <- paste0("chunk", chunk_id, "_param", i)
    # output_file <- file.path(result_dir, paste0(param_id, ".csv"))
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
      res_b <- run_batch_music(batches[[b]], param_row, batch_counter, log_file = LOG_FILE)
      combo_results[[b]] <- res_b
      batch_counter <- batch_counter + 1
    }
    
    result_df <- bind_rows(combo_results)
    write_csv(result_df, output_file)
    cat(glue("[{Sys.time()}] Finished {param_id}, saved {nrow(result_df)} rows\n"))
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

# run experiments
global_batch <- 1
for (i in seq_along(param_chunks)) {
  cat(glue("\n[{Sys.time()}] === Running chunk {i}/{length(param_chunks)} ===\n"))
  global_batch <- run_exp_chunk_music(
    music_data, 
    param_chunks[[i]], 
    chunk_id = i, 
    start_batch = global_batch, 
    log_dir = log_dir)
}

# merge all result CSVs into one final file
all_new_csvs <- list.files(list_dir, pattern = "\\.csv$", full.names = TRUE)
all_new_results <- purrr::map_dfr(all_new_csvs, read_csv, show_col_types = FALSE)

final_file <- file.path(result_dir, paste0(
  "music_scoring_sensitivity_newgrid_all_results_1000sample_", 
  format(Sys.time(), "%Y-%m-%d-%H%M%S"), ".csv"))

write_csv(all_new_results, final_file)



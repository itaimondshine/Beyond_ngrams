# 📘 Beyond N-Grams: Rethinking Evaluation Metrics and Strategies for Multilingual Abstractive Summarization

This repository supports research focused on evaluating multilingual abstractive summarization beyond traditional n-gram-based metrics (e.g., ROUGE). Our approach emphasizes **human-centric evaluation**—particularly around the dimensions of **coherence** and **completeness**—and investigates the limitations of current automatic metrics.

---

## 📁 Repository Structure

### 1. `app/`

This folder contains the **user interface** used for collecting human evaluation scores.

- Built with **Flask** and deployed using **Google Cloud Run**.
- Includes a subfolder with:
  - **Source documents**.
  - **Summaries** from two models: *Gemini* and *GPT*.
  - **Corrupted and not corrupted summaries** of the original summaries, as displayed to the users.

These were displayed to human annotators during the evaluation process.

---

### 2. `data/`

Contains the **human-annotated evaluation data**, organized by evaluation criterion:

- `coherence/` — human scores based on **coherence**.
- `completeness/` — human scores based on **completeness**.

Each folder contains a DataFrame (CSV or pickle file) with the following columns:

| Column Name                 | Description |
|----------------------------|-------------|
| `inner_index`              | Unique identifier for the data row. Useful for indexing and cross-referencing. |
| `coherence_gemini` / `completeness_gemini` | Human ratings for the (list of scores) **Gemini-generated summary**. |
| `coherence_gpt` / `completeness_gpt`       | Human ratings for the (list of scores) **GPT-generated summary**. |
| `label`                    | the gold true label from the dataset. |
| `text`                     | The original **source document** from which the summaries were generated. |
| `gemini_corrupted_summary` | The **version displayed for the user** based on 'orig_gemini_prediction' and after applied 'config' (in most of the times there was no corruption. In that case - 'orig_gemini_prediction' is Nan) |
| `gpt_corrupted_summary`    | The * **version displayed for the user** based on 'orig_gpt_prediction' and after applied 'config' (in most of the times there was no corruption. In that case - 'orig_gpt_prediction' is Nan) |
| `config`                   | Configuration for the corruption applied to the original generated summaries |
| `orig_gpt_prediction`      | The **original, uncorrupted GPT-generated summary**|
| `orig_gemini_prediction`   | The **original, uncorrupted Gemini-generated summary**. |

> 💡 *Corrupted summaries were included to test human sensitivity to quality degradation in summaries and evaluate robustness of human judgment.*


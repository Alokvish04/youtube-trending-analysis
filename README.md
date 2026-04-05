# YouTube Trending Analysis

This project analyzes YouTube trending videos and builds a machine learning model to predict whether a trending video will cross 1M views using Python, pandas, Seaborn, and scikit-learn.

## Dataset

- Source: Kaggle – "Trending YouTube Video Statistics" (US subset).
- Collected over 200+ days with data on tens of thousands of trending videos.
- Key columns: `title`, `views`, `likes`, `dislikes`, `comment_count`, `description`, `category_id`, etc. [web:2]

## Objectives

- Explore what is common among trending YouTube videos. [web:2]
- Understand engagement patterns (views, likes, dislikes, comments).
- Analyze title characteristics (length, capitalization, common words).
- Build a binary classifier to predict if a video will cross 1M views.

## Data Cleaning and Feature Engineering

- Filled missing values in `description` with empty strings.
- Removed or handled obvious anomalies when needed.
- Created additional features:
  - `contains_capitalized`: whether the title contains at least one fully capitalized word.
  - `title_length`: number of characters in the title.
  - `high_views`: 1 if `views` ≥ 1,000,000, else 0 (target for ML).

## Exploratory Data Analysis (EDA)

Some of the key insights from the dataset:

- The average number of views of a trending video is around a few million, while the median is under 1M; many trending videos do not cross the 1M mark. [web:2]
- Trending videos typically receive a large number of likes and a substantial number of comments, with likes and views highly correlated. [web:2]
- A significant fraction of trending video titles contain at least one fully capitalized word (e.g., "WHAT"). [web:2]
- Title length is usually in a moderate range (roughly 30–60 characters) for most trending videos. [web:2]
- Correlation analysis shows a strong positive relationship between views and likes, and moderate relationships with comments and dislikes. [web:2]
- A word cloud over titles highlights frequent words and themes among trending videos. [web:2]

(See the notebook for plots such as distributions, correlation heatmaps, and the word cloud.)

## Machine Learning Component

On top of the EDA, the project frames a binary classification task:

- **Goal:** Predict whether a trending video will cross a threshold of 1,000,000 views.
- **Target variable:** `high_views` (1 if views ≥ 1M, else 0).
- **Features used:**
  - `likes`
  - `dislikes`
  - `comment_count`
  - `title_length`
  - `contains_capitalized`

### Model and Evaluation

- Model: Logistic Regression (scikit-learn).
- Train/test split with stratification on the target.
- Evaluation metrics:
  - **Accuracy:** ~86.7% on the held‑out test set.
  - **Class 1 (high views) performance:**
    - Precision: 0.89  
    - Recall: 0.76  
    - F1-score: 0.82  

These results show that using simple, interpretable engagement and title features, a logistic regression classifier can effectively distinguish high‑view videos from others.

## What This Project Demonstrates

From an analysis perspective, this project shows:

- Typical engagement levels and distributions for trending videos.
- How title formatting (length and capitalization) appears in trending content.
- How strongly views relate to likes, dislikes, and comments. [web:2]

From a skills perspective, this project demonstrates that you can:

- Work end‑to‑end with a real-world dataset (loading, cleaning, feature engineering).
- Perform EDA with visualizations and summary statistics to extract insights.
- Frame a supervised ML problem, build a model, and evaluate it with appropriate metrics.
- Interpret the model’s performance and relate it to a practical question: predicting which trending videos are likely to cross 1M views.

## Tech Stack

- Python
- pandas, NumPy
- Matplotlib, Seaborn
- wordcloud
- scikit-learn

## Project Structure

- `notebooks/` – Jupyter notebook(s) with EDA and model training (add your exact filename here).
- `data/` – Place the dataset here (e.g., `USvideos.csv`).
- `README.md` – Project overview (this file).

(Adjust the folders above if your actual structure is different.)

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Alokvish04/youtube-trending-analysis.git
   cd youtube-trending-analysis
   ```
2. Download the `USvideos.csv` file from the Kaggle “Trending YouTube Video Statistics” dataset and place it in the `data/` folder (or project root, matching the notebook path). [web:5]
3. (Optional) Create and activate a virtual environment.
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Open the notebook:
   ```bash
   jupyter notebook
   ```
   and run all cells, or run the Python script if you have one for end‑to‑end execution.

## Future Work

- Try more advanced models (Random Forest, XGBoost, etc.) and compare their performance to logistic regression.
- Add richer text-based features from titles and descriptions (TF-IDF, embeddings).
- Incorporate publish time and category information for temporal and categorical analysis.
- Extend analysis to multiple countries and compare trends across regions. [web:2]

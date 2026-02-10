<h1>SalFUP</h1>
<p><strong>A Hybrid Framework for Sales Forecasting and Product Upgradation Planning</strong></p>

<hr>

<h2>Overview</h2>
<p>
SalFUP is an end-to-end hybrid framework for automated sales forecasting and
product upgradation planning. The entire pipeline is executed through a single
Jupyter notebook for reproducibility.
</p>

<h2>Key Features</h2>
<ul>
    <li>Automatic dataset download</li>
    <li>Data preprocessing and feature engineering</li>
    <li>Hybrid sales forecasting models</li>
    <li>Product upgradation impact analysis</li>
    <li>Automatic storage of results and intermediate files</li>
</ul>

<h2>System Requirements</h2>
<ul>
    <li>Python 3.10+</li>
    <li>Jupyter Notebook / JupyterLab</li>
    <li>Windows / Linux / macOS</li>
</ul>

<h2>Download Steps</h2>
<pre>
git clone &lt;repository-url&gt; SalFUP
cd SalFUP
</pre>

<h2>How to Run</h2>
<pre>
jupyter notebook runner.ipynb
</pre>

<p>
Open <code>runner.ipynb</code> and run each cell sequentially.
The notebook will automatically download the dependencies, download the dataset, train models,
and save all results locally.
</p>

<h2>Project Structure</h2>
<pre>
SalFUP/
├── main.py                     # Pipeline runner (argument-based execution)
├── utils/
│   └── data_preprocess.py      # Data & sentiment preprocessing
├── scripts/
│   ├── train_lstm.py
│   ├── train_arima.py
│   ├── train_sarima.py
│   ├── train_sentitsmixer.py
│   ├── shap_importance.py
│   ├── apply_shap_weights.py
│   ├── review_decline_analysis.py
│   └── review_growth_analysis.py
├── data/                        # Intermediate Files are stored
│   ├── monthly_grouped.csv
│   ├── monthly_grouped_shap.csv
│   └── raw_reviews.csv
├── results/                     # Results of each run are stored here
│   ├── pre_tuning/
│   └── post_tuning/
├── requirements.txt
├── runner.ipynb                 # Python code runner
└── README.md
</pre>

<h2>Pipeline Flow</h2>
<ol>
    <li>Data preprocessing & sentiment extraction</li>
    <li>Model training (LSTM, ARIMA, SARIMA, SentiTSMixer)</li>
    <li>SHAP feature importance computation</li>
    <li>SHAP-based feature reweighting</li>
    <li>Post-tuning model retraining</li>
    <li>Review analysis during sales decline</li>
    <li>Review analysis during sales growth</li>
</ol>

<h2>Outputs</h2>
<ul>
    <li>Sales forecasting results</li>
    <li>Evaluation metrics and visualizations</li>
    <li>Product upgradation insights</li>
    <li>Intermediate cached files</li>
</ul>

<h2>Notes</h2>
<ul>
    <li>No manual dataset handling required</li>
    <li>Internet connection required for dataset download</li>
</ul>

<h2>Support</h2>
<p>
If you face any issues or have questions, feel free to reach out to the authors.
</p>

</body>
</html>

# cs5293sp23-project3
## Dependencies
The following dependencies are required to run the code:

#### pypdf
#### normalize
#### os
#### json
#### argparse
#### pandas
#### scikit-learn
#### pickle
#### oblib

## Working:
#### Working of the Code

* **The clean_pdf_files()** function takes a list of PDF filenames as input.

* For each PDF file, the function extracts the text from all pages and stores it in a dictionary.

* The dictionary is converted to a pandas DataFrame with two columns: "city" and "Raw_text". 

* The **"Raw_text"** column is normalized using the normalize_corpus() function from the normalize package.

* The pre-trained machine learning model (a KMeans clustering model) is loaded from the model.joblib file using the **load()** function from the joblib package.

* The vectorizer used to transform the text data is loaded from the **vectorizer.joblib** file using the load() function from the joblib package.

* The cleaned text data is transformed using the vectorizer.

* The pre-trained model predicts the cluster ID for each city based on its cleaned text data.

* The function creates a new column in the DataFrame with the cluster ID.

* The DataFrame is saved as a TSV file named smartcity_predictions.tsv.

* The function returns the city name and the cluster ID.

* In the main block of the script, an argument parser is created to allow the user to input the names of the PDF files from the command line using the --document flag. 

* The clean_pdf_files() function is called with the input filenames, and the city name and cluster ID are printed to the console.

* The output will be a TSV file named smartcity_predictions.tsv containing the city name, the raw text extracted from the PDF, the cleaned text, and the cluster ID assigned to that city.

*** pipenv run python project3.py --document "Oklahoma.pdf"** this command is used to execute program

## Machine Learning Model

* The code uses a pre-trained machine learning model to cluster the PDF files. 
* The model is loaded from the model.joblib file, and the vectorizer used to transform the text data is loaded from the vectorizer.joblib file. 
* The model used is a KMeans clustering model, and the quality of the clustering is evaluated using the Silhouette score, Calinski-Harabasz score, and Davies-Bouldin score.





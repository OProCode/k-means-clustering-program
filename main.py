import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import Optional

def convert_xlsx_to_csv(file_path, csv_file_path, sheet, columns: Optional[None]):
    """Converts the passed xlsx file with the corresponding sheet name and data in the specifies columns to a csv

    Arguements:
        file_path {string} -- Path to xlsx file to be converted
        csv_file_path {string} -- Path to save converted csv file
        sheet {string} -- Name of the xlsx sheet to read data from
        columns {string} -- Optional arguement, reads data from specified columns in the sheet
    """

    try:
        df = pd.read_excel(file_path, sheet_name=sheet, usecols=columns)
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return

    try:
        df.to_csv(csv_file_path, index=False)
        print(f"Successfully converted {file_path} to {csv_file_path}")
    except Exception as e:
        print(f"Error writing the CSV file: {e}")

def convert_to_numpy_list(fname):
    """Converts the passed file to a NumPy array

    Arguements:
        fname {string} -- Path to file to be converted

    Returns:
        ndarray -- Organised NumPy array of data from passed file
    """

    df = pd.read_csv(fname)
    units_sold = df['Units Sold'].to_numpy()
    sales = df['Sales'].to_numpy()

    return units_sold, sales

def find_optimal_k(data_point_1, data_point_2):
    """Uses passed data to determine the optimal number of k value clusters

    Arguements:
        data_point_1 {ndarray} -- Organised NumPy array of data from passed file
        data_point_2 {ndarray} -- Organised NumPy array of data from passed file

    Returns:
        integer -- K value nuumber
    """
    
    data = np.column_stack((data_point_1, data_point_2))

    wcss_values = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss_values.append(kmeans.inertia_)

    differences = [wcss_values[i] - wcss_values[i - 1] for i in range(1, len(wcss_values))]

    optimal_k = differences.index(max(differences)) + 1

    return optimal_k

def plot_data(data_point_1, data_point_2, optimal_k):
    """Plots data into a clustered scatter graph and elbow method plot using passed in optimal k value

    Arguements:
        data_point_1 {ndarray} -- Organised NumPy array of data
        data_point_2 {ndarray} -- Organised NumPy array of data
        optimal_k {integer} -- Value determining optimal number of data clusters
    """

    data = np.column_stack((data_point_1, data_point_2))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    ax1.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k', label='Data Points')
    ax1.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red', label='Centroids')
    for i, (x, y) in enumerate(centroids):
        ax1.text(x, y, str(i + 1), ha='center', va='center', fontsize=10, color='black')
    ax1.set_title('K-means Clustering')
    ax1.set_xlabel('Units Sold')
    ax1.set_ylabel('Sales')
    ax1.legend()

    wcss_values = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss_values.append(kmeans.inertia_) 

    ax2.plot(range(1, 11), wcss_values)
    ax2.set_title('Elbow Method')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('WCSS')

    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    # First, we set our values for reading and converting the files

    xlsx_file_path = 'clustering.xlsx'
    sheet_name = 'Sheet1'
    csv_file_path = 'clustering.csv'
    columns = ['Units Sold', 'Sales']

    # Next, we run the conversion
    convert_xlsx_to_csv(xlsx_file_path, csv_file_path, sheet_name, columns)
    
    # We then run the conversion to numpy arrays and instantiated the returned values
    units_sold, sales = convert_to_numpy_list(csv_file_path)

    # Using these values, we find and instantiate the optimal number of clusters (k value)
    optimal_k = find_optimal_k(units_sold, sales)

    # Finally, we use the optimal k value along with our numpy arrays to plot the clusterd scatter graph and elbow method
    plot_data(units_sold, sales, optimal_k)

    # By looking at the elbow method, it appears to k value should be '3'. However, the optimal k returned is in fact '9'
    # In this situation, grouping the data into 3 clusters may be more appropriate than 9, which we can do like so
    plot_data(units_sold, sales, 3)

    # Whether or not we should group the data into 3 clusters or 9 clusters is a matter of debate, as either may be appropriate depeding on the circumstance



# Library for data processing
import numpy as np
import pandas as pd

# Libraries for data visualisation 
import matplotlib.pyplot as plt

# Libraries for data normalisation
from sklearn.preprocessing import MinMaxScaler

#Libraries for clustering and fitting
from sklearn.cluster import KMeans
import scipy.optimize as opt

# Module for creating upper and lower limit of forecast
import err_ranges as err


def read_data(file_names):
    '''
        Reads csv files, merges them into single dataframe. the returns orignal and transposed df
        
        Parameters:
            file_names (list): List of csv files to be merged.
        Returns:
            merged_df (pd.DataFrame): Merged dataframe from input files.
    '''
    
    
    #read each csv files and return merged dataframe
    dfs = [pd.read_csv(csv_file + '.csv') for csv_file in file_names]
    original_df = pd.concat(dfs)
    
    # transpose, rename dataframe columns, drop non-year rows and convert whole df to float64 dtypes
    transposed_df = original_df.set_index('Country Name').T
    second_column_header = transposed_df.iloc[1].tolist()
    transposed_df.columns = [ transposed_df.columns,  second_column_header ]
    transposed_df.drop(['Country Code','Indicator Name', 'Indicator Code'], axis=0, inplace=True)
    transposed_df = transposed_df.apply(pd.to_numeric, errors='coerce')
    
    return original_df, transposed_df


# read two csv files and merge them into single dataframe
df, transposed_df = read_data(['API_19_DS2_en_csv_v2_4773766', 'GDP per capita'])
row, col = df.shape 

print(f'there are {row} rows and {col} columns in the dataset')

# Explore dataframe and the accompaning climate change factors
print(df.head())
count_of_ind = df['Indicator Name'].nunique()
print('We have ' + str(count_of_ind) + ' climate change factors in this data set')
print(df['Indicator Name'].unique())


# To begin clustering we need to narrow dataset to a specific year
year = 2010


def transform(df, year):
    """
    Filter a Dataframe by a specific year, keeping only 'Country Name', 
    'Indicator Name', and the selected year columns, then it unstacks it
    by 'Country Name' and 'Indicator Name' columns.
    
    Parameters:
        dataframe (pd.DataFrame): Dataframe that needs to be filtered.
        year (int): The year to be used as a filter.
        
    Returns:
        filtered_df (pd.DataFrame): Filtered and transformed dataframe.
    """
    transformed_df = df[['Country Name', 'Indicator Name', str(year)]]
    transformed_df = transformed_df.set_index(['Country Name', 'Indicator Name']).unstack()
    transformed_df.columns = transformed_df.columns.droplevel(0)
    transformed_df = transformed_df.reset_index()
    transformed_df.columns.name = None

    return transformed_df



transformed_df = transform(df, year)

# Let's explore the tranformed dataframe  further 
print(transformed_df.head())
print(transformed_df['Country Name'].unique())

# store items in Country name column that are not countries
other_categories = [
    'Europe & Central Asia', 'East Asia & Pacific (excluding high income)', 
    'Pacific island small states', 'Middle East & North Africa (excluding high income)', 
    'Heavily indebted poor countries (HIPC)', 'Least developed countries: UN classification', 
    'Small states', 'Pre-demographic dividend', 'World', 'European Union', 
    'South Asia', 'Central Europe and the Baltics', 'Middle East & North Africa', 
    'Latin America & Caribbean', 'South Asia (IDA & IBRD)', 'IDA blend', 
    'Latin America & Caribbean (excluding high income)', 'Sub-Saharan Africa', 
    'Low & middle income', 'OECD members', 'Europe & Central Asia (IDA & IBRD countries)', 
    'IBRD only', 'Early-demographic dividend', 'Caribbean small states', 
    'Europe & Central Asia (excluding high income)', 'Middle income', 
    'Africa Western and Central', 'Post-demographic dividend', 
    'Latin America & the Caribbean (IDA & IBRD countries)', 'Low income', 
    'Africa Eastern and Southern', 'Upper middle income', 'Euro area', 
    'Late-demographic dividend', 'Lower middle income', 'Sub-Saharan Africa (IDA & IBRD countries)', 
    'East Asia & Pacific', 'IDA total', 'Sub-Saharan Africa (excluding high income)', 
    'IDA only', 'East Asia & Pacific (IDA & IBRD countries)', 
    'Middle East & North Africa (IDA & IBRD countries)', 'Fragile and conflict affected situations', 
    'Other small states', 'IDA & IBRD total'
    ]



def clean_dataframe(df, indicators):
    """
    Clean a dataframe by keeping only the specified columns and removing missing values
    and certain items that are not countries from the dataset.
    
    Parameters:
        df (pd.DataFrame): Dataframe that needs to be cleaned.
        indicators (list): List of factor for clustering.
        
    Returns:
        filtered_df (pd.DataFrame): Cleaned dataframe.
    """
    filtered_df = df[['Country Name'] + indicators]
    filtered_df = filtered_df.dropna()
    filtered_df = filtered_df[~filtered_df['Country Name'].isin(other_categories)]
        
    return filtered_df



# select factors to use in clustering
clustering_var = [
    'GDP per capita (current US$)', 'CO2 emissions from liquid fuel consumption (% of total)', 
    'Energy use (kg of oil equivalent per capita)', 'Urban population (% of total population)'
    ]


final_df = clean_dataframe(transformed_df, clustering_var)

# Now explore the final dataset ready for clustering and get min and max of variables
print(final_df.head())
print(final_df.shape)
print(final_df.describe().loc[['min', 'max'], :])


# vizualize 2 variables on scatterplot
def scatter_plot(dataframe, x_indicator, y_indicator, color = None):
    """
    Create a scatter plot of the specified indicators from the dataframe.
    
    Parameters:
        dataframe (pd.DataFrame): Dataframe that contains the indicators.
        x_indicator (str): The name of the column in the dataframe to be used as the x axis.
        y_indicator (str): The name of the column in the dataframe to be used as the y axis.
        color (str, optional): The color of the points in the scatter plot.
    """
    plt.scatter(dataframe[x_indicator], dataframe[y_indicator], color = color)
    plt.xlabel(x_indicator)
    plt.ylabel(y_indicator)
    plt.show()


# visualize data set on scatter plot before clustering
for x_indicator in clustering_var[1:]:
    scatter_plot(final_df, x_indicator, 'GDP per capita (current US$)')


def cluster_dataframe(df, n_clusters, indicators):
    """
    Apply K-Means clustering to a dataframe based on the specified indicators.
    
    Parameters:
        df (pd.DataFrame): The Dataframe to be clustered.
        n_clusters (int): The number of clusters to be created.
        indicators (List[str]): The list of indicator columns used for clustering.
    
    Returns:
        Tuple[newdf:pd.DataFrame, centroid: pd.DataFrame]: A tuple containing the clustered dataframe and the centroid dataframe.
    """
    # make a copy of the original dataframe
    newdf = df.copy()
    
    # initiate the k-means model
    km = KMeans(n_clusters=n_clusters)
    
    # fit the model to the data and predict the clusters
    cluster_group = km.fit_predict(newdf[indicators])
    
    # add the cluster predictions to the dataframe
    newdf['cluster'] = cluster_group
    
    # create a dataframe for the cluster centroids
    centroid = pd.DataFrame(km.cluster_centers_, columns = indicators)
    return newdf, centroid



#cluster dataset into 3 seperate groups using the selected factors
clustered_df, centroid = cluster_dataframe(final_df, 3, clustering_var)
# clustered_df.to_csv(f'Cluster year {selected_year}.csv')


def plot_cluster_scatter(df, centroids, n_clusters, colors, x_col, y_col, year):
    """
    Plots clusters of a given dataframe on a scatter plot using different colors.

    Parameters:
        dataframe (pd.DataFrame): The clustered dataframe to be visualized.
        centroids (pd.DataFrame): The centroid dataframe used for the clustering.
        n_clusters (int): The number of clusters in the dataframe.
        colors (list): A list of colors to use for the clusters. Should have a length equal to num_clusters.
        x_col (str): The name of the column to use for the x-axis.
        y_col (str): The name of the column to use for the y-axis.
        year (int): The year the data represents

    """
    for i in range(n_clusters):
        cluster = df[df['cluster'] == i]
        plt.scatter(cluster[x_col], cluster[y_col], color=colors[i], label = f'cluster {map_color[colors[i]]}')
    plt.scatter(centroids[x_col], centroids[y_col], color='purple', marker='x', label='centroid')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.title('Country Clusters for '  + str(year))
    plt.show()

    

colors = ['green', 'red', 'black']
map_color = {
    'green': 0, 'red' : 1, 'black':2
    }

# display cluster for gdp and other variables on scattered plot
for x_indicator in clustering_var[1:]:
    plot_cluster_scatter(clustered_df, centroid, 3, colors, x_indicator,  'GDP per capita (current US$)', year)


#find optimal number of clusters
def get_optimal_k(df, indicators):
    '''
        uses the elbow method to find optimal number of clusters 
        
        Args:
            df => pandas.Dataframe, clustered dataframe to be visualize
            centroid_df => pandas.Dataframe, dataframe to be transformed
            k => int, number of clusters
            x => first indicator
            y => second indicator
        Returns:
             grapgh => scatterplot
    '''
    sse = []
    cluster_range = range(1,11)
    for n_clusters in cluster_range:
        km = KMeans(n_clusters=n_clusters)
        km.fit(df[indicators])
        sse.append(km.inertia_)
    
    plt.xlabel('K')
    plt.ylabel('Sum of squared error')
    plt.plot(cluster_range,sse)
    plt.show()


get_optimal_k(final_df, clustering_var)


def compute_cluster_statistics(df):
    """
    Compute statistics for each cluster in the dataframe, such as mean, count, and median.
    
    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the cluster column.
        
    Returns:
        statistics_dataframe (pd.DataFrame): A dataframe containing the statistical summary of the different clusters.    
    """
    statistics_dataframe = df.groupby('cluster').agg(['mean', 'count', 'median'])
    return statistics_dataframe


cluster_statistics = compute_cluster_statistics(clustered_df)
print(cluster_statistics)

# lets view the content of each cluster to pick a country we would predict futuristic value for
cluster_2 = clustered_df[clustered_df['cluster'] == 2]
print(cluster_2.head())

cluster_1 = clustered_df[clustered_df['cluster'] == 1]
print(cluster_1.head())


# explore the clustering of countries 20 years ago by repeating the process for 1990
year = 1990
colors = ['red', 'black', 'green']
transformed_df = transform(df, year)
final_df = clean_dataframe(transformed_df, clustering_var)

clustered_df, centroid = cluster_dataframe(final_df, 3, clustering_var)
# clustered_df.to_csv(f'Cluster year {year}.csv')
for x_indicator in clustering_var[1:]:
    plot_cluster_scatter(clustered_df, centroid, 3, colors, x_indicator,  'GDP per capita (current US$)', year)
    
cluster_statistics = compute_cluster_statistics(clustered_df)

cluster_2 = clustered_df[clustered_df['cluster'] == 2]
print(cluster_2.head())

cluster_1 = clustered_df[clustered_df['cluster'] == 1]
print(cluster_1.head())


def prepare_fit_data(df, selected_country, selected_indicator):
    """
    Prepare the dataframe for fitting by filtering for specific country and indicator and 
    formatting the dataframe for fitting.

    Parameters:
        df (pd.DataFrame): The original dataframe.
        selected_country (str): The name of the country of interest.
        selected_indicator (str): The indicator of interest.

    Returns:
        fit_data (pd.DataFrame): A dataframe ready for fitting.
    """
    fit_data = df[(df['Country Name'] == selected_country) & (df['Indicator Name'] == selected_indicator)]
    fit_data = fit_data.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis=1).T
    fit_data = fit_data.reset_index()
    fit_data.columns = ['Year', selected_indicator]
    fit_data = fit_data.dropna()
    fit_data['Year'] = pd.to_numeric(fit_data['Year'])
    
    return fit_data



# Qatar GDP trend over the years
China_GDP = prepare_fit_data(df, 'China', 'GDP per capita (current US$)')
print(China_GDP.head())


def exponential(t, n0, g):
    '''Calculates exponential function with scale factor n0 and growth rate g.'''
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f


def logistic(t, n0, g, t0):
    '''Calculates the logistic function with scale factor n0 and growth rate g'''
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


def fit_curve(df, indicator, fit_func, initial_guess):
    """
    Fit a given function to a dataset.
    
    Parameters:
        df (pd.DataFrame): The original dataframe.
        indicator (str): The indicator of interest.
        fit_func (callable): The function to fit.
        initial_guess (Tuple[float]): The initial guess of coefficients.
    
    Returns:
        Tuple[np.ndarray, np.ndarray] : Tuple containing the fitted parameters and covariance matrix.
    """
    # new_df = df.copy()
    # new_df['Year'] = pd.to_numeric(new_df['Year'])
    fitted_params, covariance = opt.curve_fit(fit_func, df['Year'], df[indicator], p0=initial_guess)
    return fitted_params, covariance



def plot_fitted_function(df, fit_function, indicator):
    '''
        plots fitting function and the original dataset
        
        Parameters:
            df (pd.DataFrame): The original dataframe.
            fit_function (callable): The function to fit.
            indicator (str): The indicator of interest.
    '''
    newdf = df.copy()
    newdf['Year'] = pd.to_numeric(newdf['Year'])
    newdf['fit'] = fit_function(newdf['Year'], *param)
    newdf.plot('Year', [indicator, 'fit'])
    plt.title('GDP per Capita trend for China')
    plt.show()


# initial guess for exp function coefficients
exp_p0 = (73233967692.102798, 0.03)

# fit exp function
param, covar = fit_curve(China_GDP, 'GDP per capita (current US$)', exponential, exp_p0 )

# plot fitting and original data set
plot_fitted_function(China_GDP, exponential, 'GDP per capita (current US$)')


# initial guess for logistic function coefficients
log_p0 = (3e12, 0.03, 2000.0)

param, covar = fit_curve(China_GDP, 'GDP per capita (current US$)', logistic, log_p0 )
plot_fitted_function(China_GDP, logistic, 'GDP per capita (current US$)')


# clearly the exponential has a beter fitting so lets predict future values



def plot_future_value(df, start_year, end_year, fit_function, y_var, param, sigma):
    """
    Plots the future values of a variable using an exponential function and error ranges.
    
    Parameters:
       df (dataframe): Dataframe containing the data
        start_year (int): Starting year for the forecast
        end_year (int): Ending year for the forecast
        fit_function (callable): fitting function for forecast
        y_var (str): Column name of the variable to be plotted
        param (tuple): Parameters of the function
        sigma (tuple): Standard deviation of the parameters
    """
    newdf = df.copy()
    newdf['Year'] = pd.to_numeric(newdf['Year'])
    year = np.arange(start_year, end_year)
    forecast = fit_function(year, *param)
    
    low, up = err.err_ranges(year, fit_function, param, sigma)
    
    plt.figure()
    plt.plot(newdf['Year'], newdf[y_var], label=y_var)
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel('Year')
    plt.ylabel(y_var)
    plt.title('GDP per Capita forecast for China')
    plt.legend()
    plt.show()

param, covar = fit_curve(China_GDP, 'GDP per capita (current US$)', exponential, exp_p0 )
sigma = np.sqrt(np.diag(covar))

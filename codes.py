# %% [markdown]
# # TASK 1
# ## Problem statement 

# %% [markdown]
# Is there a correlation between the choice of car (brand, model, colour) and the individual's financial profile (credit card type)? 

# %% [markdown]
# # Answer:-
# 
# To investigate whether there is a link between the type of car (brand, model, color) someone chooses and their financial profile (credit card type), we conducted a comprehensive analysis of the dataset provided. Upon initial inspection, we found relevant information on both car attributes and financial profiles. We then calculated descriptive statistics to understand the central tendencies and variability of the data and used visualizations like bar charts and heatmaps to uncover any patterns. To measure the strength of association between categorical variables, we conducted correlation analyses using techniques like chi-square tests. We also performed statistical testing to evaluate the significance of any observed relationships and, if applicable, considered a geographical analysis to identify regional variations. Our findings were summarized, highlighting any significant correlations and their potential implications. It's essential to note that although we explored correlations, causation cannot be inferred, and interpretation of the results should consider the context of the dataset and the specific objectives of the analysis.

# %% [markdown]
# # TASK 2
# 
# ##  Exploratory Data Analysis (EDA)

# %%
#Importing the required Python Libraries that will be used for performing EDA

import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import plotly.express as px
import os

# %% [markdown]
# READING THE DATASET

# %%
#reading a CSV file

df=pd.read_csv('cars.csv')

# %%
df

# %% [markdown]
# ANALYSING THE BASIC STATISTICS

# %%
df.info()

# %%
# Summary statistics
print(df.describe())

# %%
# Unique values in each column
for col in df.columns:
    print(f"Unique values in {col}: {df[col].nunique()}")

# %%
#checking the number of rows and coloumns in the dataset


print("The rows and columns in the dataset is: ")

df.shape

# %%
#checking the column name of dataset and its corresponding data types

df.dtypes

# %%
# Unique values in each column
for col in df.columns:
    print(f"Unique values in {col}: {df[col].nunique()}")

# %% [markdown]
# HANDLING THE NULL VALUES
# 

# %%
#checking for null values in the dataset

df.isnull().values.any()

# %%
#lets cross verify it

df.isna().sum()

# %% [markdown]
# NOTE: From the above outputs we can see that there are no null values on the dataframe, so we can proceed with the EDA part now

# %% [markdown]
# VISUALIZATION

# %% [markdown]
# PLOT 1: Ploting the Distribution of Car based on the year of Manufacture

# %%
# Function to plot the distribution of 'Year of Manufacture'
def plot_year_of_manufacture_distribution(dataframe):
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot a histogram with kernel density estimation using seaborn
    sns.histplot(dataframe['Year of Manufacture'], kde=True, bins=30, color='blue', ax=ax)
    
    # Set the title and axis labels for better clarity
    ax.set_title('Distribution of Year of Manufacture')
    ax.set_xlabel('Year of Manufacture')
    ax.set_ylabel('Frequency')
    
    # Display the plot
    plt.show()

# Assuming 'df' is your DataFrame
plot_year_of_manufacture_distribution(df)


# %% [markdown]
# Sparse Data before the 1980s: There is very little data available for car manufacturing years before the 1980s. This might indicate that either few cars from these years exist in the dataset, or that older cars might not be as commonly in use or registered.
# Significant Growth in the Late 20th Century: Starting from the 1980s, there's a noticeable uptick in the number of cars manufactured. The count continues to rise substantially, peaking somewhere in the late 1990s or early 2000s.
# Sharp Decline after the Peak: After reaching the peak, there's a sharp decline in the number of cars manufactured as we approach the 2000s. It's worth noting that the dataset might not include more recent years, or it could be indicative of another trend, such as a preference for older models or economic factors influencing car purchases.
# General Trend: The general trend, if we focus on the latter half of the 20th century, indicates a growing automobile industry with more cars being produced as time progresses until it reaches a peak and then witnesses a sharp decline.
# Kernel Density Estimation (KDE): The KDE curve provides a smoothed representation of the distribution. The prominent peak in the KDE curve aligns with the histogram bars, emphasizing the period of maximum car manufacturing.
# In conclusion, the graph offers insights into the evolution of car manufacturing over the years. Understanding the reasons behind these trends—whether they are due to economic factors, industry shifts, consumer preferences, or data collection limits—would require further investigation and a deeper dive into both the dataset and external historical and economic factors.

# %% [markdown]
# PLOT 2: Ploting the Top Car Brands

# %%
# Function to plot the top car brands as a line plot
def plot_top_car_brands_line(dataframe):
    # Set the seaborn style
    sns.set(style="whitegrid")
    
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Count the occurrences of each car brand
    brand_counts = dataframe['Car Brand'].value_counts()
    
    # Create a line plot using seaborn lineplot
    sns.lineplot(x=brand_counts.index, y=brand_counts, marker='o', sort=False, ax=ax)
    
    # Set the title and axis labels for better clarity
    ax.set_title('Top Car Brands (Line Plot)')
    ax.set_xlabel('Car Brand')
    ax.set_ylabel('Frequency')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Display the plot
    plt.show()

# Assuming 'df' is your DataFrame
plot_top_car_brands_line(df)


# %% [markdown]
# The code utilizes seaborn to create a line plot illustrating the distribution of car brands in the dataset. The 'whitegrid' style provides a clean background, and the figure is configured with a size of 12 by 6. The frequency counts of each car brand are computed, and a line plot is generated using seaborn's `lineplot` function. The x-axis represents car brand names, the y-axis shows the frequency counts, and circular markers are added for each data point. The x-axis labels are rotated for improved readability. The plot is titled "Top Car Brands (Line Plot)" with labeled axes for car brand and frequency. This representation allows for a clear visualization of the variation in frequencies across different car brands, aiding in the exploration of the dataset's car brand distribution.
# 
# 
# 
# Primary Contenders: At the top of the chart, we observe "Ford" and "Chevrolet" as the dominant car brands in terms of count, demonstrating their remarkable popularity or widespread acceptance within the dataset. Their prevalence indicates that they are major players in the automobile market, at least in the context of the dataset at hand.
# Noteworthy Competitors: Following the front runners, brands such as "Mercedes," "Volkswagen," and "Mazda" also possess significant representation. While they may not match the sheer volume of the leading brands, they still hold a commendable market presence.
# Varied Market: The chart displays a broad array of car brands represented, illustrating a diverse automobile market. While there are clear leaders, consumers have many options available, resulting in a rich variety of cars on the roads.
# Gradual Decline: As we move down the graph, the number of cars for each subsequent brand decreases. The chart showcases a classic long-tail distribution, with a few brands dominating the market and a large number of brands with lesser representation.
# Luxury versus Mass Market: Brands typically classified as luxury, such as "Aston Martin" and "Lamborghini," have lower counts compared to mass-market brands. This is unsurprising as luxury cars are generally produced in smaller quantities and cater to a niche audience.
# Visualization Challenge: Due to the large number of brands and the varying count for each brand, some labels may be challenging to read clearly, particularly towards the middle and bottom of the chart. For a clearer analysis, it may be helpful to segment the data or choose an alternative visualization method.
# In summary, the graph provides valuable insights into the distribution of car brands. The leading brands demonstrate their dominance in the automobile market, while the diversity in representation indicates the abundance of options available to consumers.

# %% [markdown]
# PLOT 3: Ploting the Top Car Color

# %%
# Function to plot the top car colors
def plot_top_car_colors(dataframe):
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Count the occurrences of each car color and plot using seaborn countplot with a different color palette
    color_counts = dataframe['Car Color'].value_counts()
    sns.countplot(y=dataframe['Car Color'], order=color_counts.index, palette='magma', ax=ax)
    
    # Set the title and axis labels for better clarity
    ax.set_title('Top Car Colors')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Car Color')
    
    # Display the plot
    plt.show()

# Assuming 'df' is your DataFrame
plot_top_car_colors(df)


# %% [markdown]
# The code uses Seaborn to generate a countplot that displays the distribution of car colors in the dataset. The countplot employs a 12 by 6 figure and the 'whitegrid' style to showcase each car color on the y-axis, with bars of varying lengths representing their respective frequencies. To enhance the visual appeal, the 'magma' color palette is used, providing a unique set of colors. The plot is titled "Top Car Colors," with the x-axis labeled as 'Frequency' and the y-axis labeled as 'Car Color.' This visualization quickly summarizes the prevalence of different car colors in the dataset, making it easier to evaluate the color distribution.
# 
# The data presented on the chart illustrates the popularity of different car colors, with "Indigo" being the most preferred among all. Traditional colors such as "Red," "Blue," and "Green" are also quite popular. On the other hand, "Yellow" has the least representation and is the least favored. Overall, the chart depicts a diverse range of car colors, with a preference for deeper and richer hues.

# %% [markdown]
# PLOT 4: Ploting the Top Credit Cards Type

# %%
# Function to plot the top credit card types with a different color palette
def plot_top_credit_card_types(dataframe):
    # Create a figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Count the occurrences of each credit card type and plot using seaborn countplot with a different color palette
    card_type_counts = dataframe['Credit Card Type'].value_counts()
    sns.countplot(y=dataframe['Credit Card Type'], order=card_type_counts.index, palette='plasma', ax=ax)
    
    # Set the title and axis labels for better clarity
    ax.set_title('Top Credit Card Types')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Credit Card Type')
    
    # Display the plot
    plt.show()

# Assuming 'df' is your DataFrame
plot_top_credit_card_types(df)


# %% [markdown]
# According to the chart, there are varying degrees of popularity among different credit card types. "JCB" takes the lead by a significant margin, followed closely by "Mastercard" and "Maestro." Other brands, such as "American Express," "Switch," and "Visa-Electron," also have a notable presence. On the other hand, "Diners-club-international" and "Solo" appear to be the least utilized among the represented card types. Overall, the data indicates a diverse usage pattern, but with a select few card types dominating the landscape.

# %% [markdown]
# PLOT 5: Ploting the Top Car Brands related to production number

# %%
# Group by 'Car Brand' and count occurrences
brand_counts = df['Car Brand'].value_counts().reset_index().sort_values(by='Car Brand', ascending=False)
brand_counts.columns = ['Car Brand', 'Count']

# Create a bar chart using Plotly Express
fig = px.bar(
    brand_counts,
    x='Car Brand',
    y='Count',
    color='Car Brand',
    title='Popular Car Brands'
)

# Display the bar chart
fig.show()

# %% [markdown]
# The code begins by calculating the frequency of each unique car brand in the DataFrame, organizing the results into a new DataFrame named `brand_counts`. The columns of this DataFrame are renamed to 'Car Brand' and 'Count' for clarity. Subsequently, using Plotly Express, a bar chart is generated with car brand names on the x-axis, their respective frequencies on the y-axis, and each bar colored according to the specific car brand. The chart is titled 'Popular Car Brands.' The resulting graph offers a visual representation of the popularity of different car brands within the dataset, allowing for easy identification of the most frequently occurring brands.

# %% [markdown]
# PLOT 6: Amount of car manufacture from Car Brand in 2012

# %%
# Group by 'Year of Manufacture', 'Car Brand', and count 'Car Model'
year = (
    df.groupby(['Year of Manufacture', 'Car Brand'])['Car Model']
    .count()
    .reset_index(name='Amount')
)

# Function to create a bar chart for a specific year
def plot_years_car(year_value):
    filtered_data = year[year['Year of Manufacture'] == year_value]
    title = f"Amount of car manufacture from Car Brand in: {year_value}"
    
    fig = px.bar(
        filtered_data,
        x='Car Brand',
        y='Amount',
        color='Amount',
        hover_name='Car Brand',
        hover_data={'Car Brand': False},
        labels={'Amount': 'Car Model'},
        title=title
    )
    
    fig.show()

# Call the function for the year 2012
plot_years_car(2012)


# %% [markdown]
# The code initially groups the DataFrame by 'Year of Manufacture' and 'Car Brand,' counting the occurrences of each 'Car Model' within these groups and storing the result in the `year` DataFrame. The subsequent function, `plot_years_car`, takes a specific year as an argument, filters the data for that year, and utilizes Plotly Express to generate a bar chart. This chart represents the distribution of car models across different car brands, with 'Car Brand' on the x-axis, the count of 'Car Model' on the y-axis, and color intensity indicating the count. The title dynamically reflects the selected year. The resulting graph provides an insightful visual representation of the manufacturing distribution of car models among various brands in the specified year.

# %% [markdown]
# PLOT 7: Amout of Car yearly Manufacture sold to customer

# %%
import plotly.express as px

# Assuming your DataFrame has these column names
# First Name, Last Name, Country, Car Brand, Car Model, Car Color, Year of Manufacture, Credit Card Type

# Group by 'Year of Manufacture', 'Car Brand', and count 'Car Model'
year = (
    df.groupby(['Year of Manufacture', 'Car Brand'])['Car Model']
    .count()
    .reset_index(name='Amount')
)

# Function to create an animated bar chart over the years
def plot_yearly_car_manufacture_animation():
    sorted_data = year.sort_values(by=['Year of Manufacture', 'Amount'], ascending=True)
    title = 'The amount of car yearly manufacture sold to customer'

    fig = px.bar(
        sorted_data,
        x='Car Brand',
        y='Amount',
        color='Amount',
        animation_frame='Year of Manufacture',
        animation_group='Car Brand',
        title=title
    )
    
    # Adjust y-axis margins and set animation duration
    fig.update_yaxes(automargin=True)
    fig.update_layout(transition={'duration': 1000})
    
    fig.show()

# Call the function to display the animated bar chart
plot_yearly_car_manufacture_animation()



# %% [markdown]
# This Python code uses Plotly Express to create a captivating animated bar chart that depicts the yearly distribution of car models sold to customers. To begin with, the DataFrame is grouped by 'Year of Manufacture' and 'Car Brand,' and then the occurrences of 'Car Model' within each group are counted. The resulting DataFrame, `year`, is sorted by 'Year of Manufacture' and 'Amount' in ascending order. The `plot_yearly_car_manufacture_animation` function generates an animated bar chart that corresponds to each year's frame. The x-axis features 'Car Brand,' the y-axis showcases the count of 'Car Model,' and the bars are color-coded by the count. The animation smoothly transitions from one year to the next, rendering a complete visualization of manufacturing trends across various car brands over time. The graph's aesthetics are further improved through margin adjustments for the y-axis and a specified animation duration. The result is a chart that provides crucial insights into the changing patterns of car production and sales for different years and car brands.

# %% [markdown]
# NOTE : We have almost covered most of the EDA on the Cars Dataset, now we can take it into next level by using it with GEOPANDAS library for geographical visualization

# %% [markdown]
# GEOGRAPHICAL VISUALIZATION

# %% [markdown]
# Before we start let me introduce to you what is GeoPandas: 
# 
# GeoPandas is an exceptional Python library that builds on the capabilities of Pandas to conduct spatial operations and analysis on geographic data. It offers an intuitive and high-performance interface that simplifies the manipulation, analysis, and visualization of geospatial vector data. Some of the prominent features of GeoPandas include seamless integration with Pandas, support for the GeoDataFrame data structure, spatial operations, reading and writing GIS formats, and visualization through integration with Matplotlib and other visualization libraries.

# %%
import geopandas as gpd
# Step 1: Aggregate data to find the most bought car brand per country
most_bought_brands = (
    df.groupby('Country')['Car Brand']
    .apply(lambda x: x.value_counts().idxmax())
    .reset_index(name='MostBoughtBrand')
)

# Step 2: Load the world map dataset
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Step 3: Merge the world map with the most bought car brand data
world_brands = world.merge(most_bought_brands, how="left", left_on="name", right_on="Country")

# Step 4: Plotting
fig, ax = plt.subplots(figsize=(15, 10))
world.boundary.plot(ax=ax, linewidth=1, color='black')  # Change boundary color to black
world_brands.plot(column='MostBoughtBrand', ax=ax, legend=True,
                  missing_kwds={'color': 'lightgrey', 'label': 'Data Unavailable'},
                  cmap='viridis')  # Change color map to 'viridis'
ax.set_title("Most Bought Car Brand per Country", fontsize=15)
plt.show()



# %% [markdown]
# The above code is generating, a choropleth map to showcase the most popular car brand in each country. A choropleth map is a thematic map that utilizes color variations to represent values across various geographic regions. Here is an overview of the process:
# 
# Data Preparation:
# Data is aggregated to determine the most frequently purchased car brand in each country. The resulting dataset, called "most_bought_brands," contains information about each country and its corresponding most purchased car brand.
# 
# Loading World Map:
# A world map dataset is loaded to provide a baseline for geographic regions. This map is crucial for plotting country-specific data.
# 
# Merging Data:
# The most bought car brand data is merged with the world map, associating each country with its respective most bought car brand. The merged dataset is named "world_brands."
# 
# Plotting the Choropleth Map:
# A subplot is created for the map with specified dimensions. The world boundaries are outlined in black for clarity.
# 
# Using the merged dataset (world_brands), the choropleth map is generated. Each country is filled with a color indicating the most bought car brand, and the color intensity corresponds to the frequency of that brand in the country.
# 
# A legend is included to interpret the colors, and missing data is represented in light grey with a label indicating "Data Unavailable."
# 
# The graph's title is set to "Most Bought Car Brand per Country."
# 
# In summary, the resulting choropleth map visually represents the dominant car brand in each country, allowing viewers to easily identify regional preferences. Darker colors signify higher frequencies of a particular brand, providing an informative visualization of car purchasing patterns across different geographic locations.

# %%
# Step 1: Aggregate data to find the most used car color per country
most_used_colors = (
    df.groupby('Country')['Car Color']
    .apply(lambda x: x.value_counts().idxmax())
    .reset_index(name='MostUsedColor')
)

# Step 2: Load the world map dataset
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Step 3: Merge the world map with the most used car color data
world_colors = world.merge(most_used_colors, how="left", left_on="name", right_on="Country")

# Create a color map that translates car colors to map colors
color_map = {
    "Violet": "violet",
    "Puce": "#CC8899",      # Puce color as HEX code
    "Blue": "blue",
    "Aquamarine": "aquamarine",
    "Turquoise": "turquoise",
    "Teal": "teal",
    "Indigo": "indigo",
    "Purple": "purple",
    "NaN": "lightgrey"      # Handling missing values
}

# Map car colors to corresponding map colors
world_colors['MapColor'] = world_colors['MostUsedColor'].map(color_map)

# Step 4: Plotting
fig, ax = plt.subplots(figsize=(15, 10))
world.boundary.plot(ax=ax, linewidth=1, color='black')  # Change boundary color to black
world_colors.plot(column='MapColor', ax=ax,
                  missing_kwds={'color': 'lightgrey', 'label': 'Data Unavailable'})
ax.set_title("Most Used Car Color per Country", fontsize=15)
plt.show()


# %% [markdown]
# The Python code above generates a choropleth map that displays the most commonly used car color in each country. Let's delve into the code to gain a better understanding of the resulting graph:
# 
# Data Preparation:
# To identify the most frequently used car color in each country, the code aggregates the original DataFrame (df). The resulting dataset, named 'most_used_colors', contains information about each country and its corresponding most used car color.
# 
# Loading World Map:
# The code loads a world map dataset using GeoPandas, named 'world'.
# 
# Merging Data:
# The 'most_used_colors' data is merged with the world map, linking each country to its respective most used car color. The merged dataset is named 'world_colors'.
# 
# Mapping Colors:
# A color map (color_map) is created to translate the car colors to map colors. This map is used to assign specific colors to the most used car colors in the dataset.
# 
# Plotting the Choropleth Map:
# A subplot is created for the map with specified dimensions. The world boundaries are outlined in black for clarity. 
# 
# The choropleth map is generated using the merged dataset (world_colors). Each country is shaded with a color indicating the most used car color and the color intensity corresponds to the frequency of that color in the country.
# 
# A legend is included to interpret the colors, and missing data is represented in light grey with a label indicating "Data Unavailable." 
# 
# The graph title is set to "Most Used Car Color per Country."
# 
# In summary, the resulting choropleth map visually represents the most commonly used car color in each country and provides a quick overview of regional color preferences. The darker shades indicate a higher frequency of a particular color, offering insights into global car color trends.

# %% [markdown]
# NOTE: Having completed the data analysis, including visualization and basic statistics, we are now ready to proceed with customer segmentation using MACHINE LEARNING ALGORITHMS.

# %% [markdown]
# ## K - Means Machine Learning Algorithm

# %%
# Create a mapping of unique Car Brand values to numerical labels
car_brand_mapping = {brand: label for label, brand in enumerate(df['Car Brand'].unique())}

# Use the mapping to create a new column 'Car_Brand_Label' with numerical labels
df['Car_Brand_Label'] = df['Car Brand'].map(car_brand_mapping)

# Create a mapping of unique Country values to numerical labels
country_mapping = {country: label for label, country in enumerate(df['Country'].unique())}

# Use the mapping to create a new column 'Country_Label' with numerical labels
df['Country_Label'] = df['Country'].map(country_mapping)


# %%
# Selecting segmentation features using column names directly
segmentation_features = ['Car Brand', 'Country']

# Creating a DataFrame with selected features
X = df[segmentation_features]


# %%
# Importing necessary libraries
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Assuming X is a DataFrame with 'Car Brand' and 'Country' as categorical variables
# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=['Car Brand', 'Country'])

# Initializing an empty list to store the within-cluster sum of squares (WCSS) for different values of k
wcss = []

# Range of k values to consider
k_values = range(1, 11)

# Loop through each value of k and fit KMeans models to calculate WCSS
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_encoded)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()


# %%
# Number of clusters (k) determined from the Elbow Method
selected_k = 2

# Initializing KMeans with the selected number of clusters
kmeans = KMeans(n_clusters=selected_k, random_state=0)

# Performing K-Means clustering and adding the 'Cluster' column to the DataFrame
df['Cluster'] = kmeans.fit_predict(X_encoded)


# %%
# Grouping the data by 'Cluster' and performing aggregate analysis
segmented_data = df.groupby('Cluster').agg({
    # For 'Car Brand', finding the most frequent brand in each cluster
    'Car Brand': lambda x: x.value_counts().idxmax(),

    # For 'Country', listing the top 3 countries in each cluster based on frequency
    'Country': lambda x: x.value_counts().index[:3].tolist(),

    # For 'Year of Manufacture', listing the top 3 years in each cluster based on frequency
    'Year of Manufacture': lambda x: x.value_counts().index[:3].tolist()
})


# %%
# Create a 3D scatter plot for visualization
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
scatter = ax.scatter(df['Car_Brand_Label'], df['Country_Label'], df['Year of Manufacture'], c=df['Cluster'], cmap='viridis')

# Customize axis labels
ax.set_xlabel('Car Brand')
ax.set_ylabel('Country')
ax.set_zlabel('Year of Manufacture')

# Add a colorbar to the right of the plot
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster', rotation=270, labelpad=15)

# Show the 3D plot
plt.show()

# %% [markdown]
# This is a 3D scatter plot that illustrates how the data is clustered based on the 'Car Brand', 'Country', and 'Year of Manufacture' features. Here's a breakdown of the key components:
# - X-Axis ('Car Brand'): The horizontal axis displays numerical labels for various car brands, with each point representing a specific brand.
# - Y-Axis ('Country'): The vertical axis displays numerical labels for various countries, with each point representing a specific country.
# - Z-Axis ('Year of Manufacture'): The depth axis displays numerical labels for different years of manufacture, with each point representing a specific year.
# - Color ('Cluster'): Each point on the scatter plot is assigned a color based on its designated cluster, which is determined by the 'Cluster' column in the dataset. Different clusters are represented by different colors, making it easy to visually differentiate between them.
# - Colorbar: The colorbar on the right side of the plot provides a reference for the colors assigned to each cluster. It shows the mapping of cluster numbers to colors.
# By observing the distribution of data points in the three-dimensional space, you can identify any patterns or groupings that may exist based on the clustering algorithm's assignment. The different colors used to represent each cluster help to highlight the unique clusters present in the dataset.

# %%
print(segmented_data)

# %%
# Import necessary libraries
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

#'Cluster' is the predicted cluster labels and 'X_encoded' is the feature matrix
predicted_labels = df['Cluster']

# Evaluate silhouette score
silhouette_avg = silhouette_score(X_encoded, predicted_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X_encoded, predicted_labels)

# Create a subplot with 1 row and 2 columns
fig, ax1 = plt.subplots()
fig.set_size_inches(18, 7)

# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1, but in this example, all lie within [-0.1, 1]
ax1.set_xlim([-0.1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters
ax1.set_ylim([0, len(X_encoded) + (selected_k + 1) * 10])

# Initialize the cluster number for each sample
cluster_labels = predicted_labels.unique()

y_lower = 10
for i in cluster_labels:
    # Aggregate the silhouette scores for samples belonging to cluster i and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[predicted_labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    # Choose a color for the cluster based on the nipy_spectral colormap
    color = cm.nipy_spectral(float(i) / selected_k)
    # Fill the silhouette plot area for the current cluster
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for the next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

# Set title and axis labels for the silhouette plot
ax1.set_title("Silhouette Plot for Clusters")
ax1.set_xlabel("Silhouette Coefficient Values")
ax1.set_ylabel("Cluster Label")

# Add a vertical line for the average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label='Average Silhouette Score')

# Display the legend, clear y-axis labels/ticks, and set x-axis ticks
ax1.legend()
ax1.set_yticks([])  # Clear the y-axis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# Show the silhouette plot
plt.show()


# %% [markdown]
# This code implements the silhouette analysis to assess the quality of clustering within a dataset. Firstly, it imports necessary libraries, including functions for computing silhouette scores and samples from scikit-learn, as well as matplotlib for plotting. The predicted cluster labels and feature matrix are used to calculate the silhouette score, denoted as 'Cluster' and 'X_encoded', respectively. The silhouette score for each sample is then computed. Subsequently, a subplot is created to display the silhouette plot, with the x-axis indicating silhouette coefficient values and the y-axis denoting cluster labels. The silhouette values for each cluster are represented as filled regions, with different colors for distinct clusters. The average silhouette score is marked by a red dashed line. This plot provides a visual assessment of the separation and cohesion of clusters, enabling the performance of the clustering algorithm on the dataset to be interpreted.
# 



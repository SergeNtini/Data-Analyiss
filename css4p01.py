
import pandas as pd
from collections import Counter
import numpy as np 


# Author: SERGE NTITI
# Campus: VUT

# Load the dataset
df = pd.read_csv('/path/to/movie_dataset.csv')

# Clean and prepare the dataset
## Rename columns to remove spaces
df.columns = df.columns.str.replace(' ', '_').str.lower()

# Questions and Answers

## 1. What is the highest rated movie in the dataset?
highest_rated_movie = df.loc[df['rating'].idxmax(), ['title', 'rating']]
print(f"Highest rated movie: {highest_rated_movie}")

## 2. What is the average revenue of all movies in the dataset?
average_revenue = df['revenue_(millions)'].dropna().mean()
print(f"Average revenue of all movies: ${average_revenue:.2f} million")

## 3. What is the average revenue of movies from 2015 to 2017 in the dataset?
average_revenue_2015_2017 = df[(df['year'] >= 2015) & (df['year'] <= 2017)]['revenue_(millions)'].dropna().mean()
print(f"Average revenue of movies from 2015 to 2017: ${average_revenue_2015_2017:.2f} million")

## 4. How many movies were released in the year 2016?
movies_2016_count = df[df['year'] == 2016].shape[0]
print(f"Movies released in 2016: {movies_2016_count}")

## 5. How many movies were directed by Christopher Nolan?
nolan_movies_count = df[df['director'] == 'Christopher Nolan'].shape[0]
print(f"Movies directed by Christopher Nolan: {nolan_movies_count}")

## 6. How many movies in the dataset have a rating of at least 8.0?
high_rating_movies_count = df[df['rating'] >= 8.0].shape[0]
print(f"Movies with a rating of at least 8.0: {high_rating_movies_count}")

## 7. What is the median rating of movies directed by Christopher Nolan?
nolan_median_rating = df[df['director'] == 'Christopher Nolan']['rating'].median()
print(f"Median rating of Christopher Nolan movies: {nolan_median_rating}")

## 8. Find the year with the highest average rating.
yearly_average_rating = df.groupby('year')['rating'].mean()
highest_average_rating_year = yearly_average_rating.idxmax()
print(f"Year with the highest average rating: {highest_average_rating_year}")

## 9. What is the percentage increase in number of movies made between 2006 and 2016?
movies_2006_count = df[df['year'] == 2006].shape[0]
percentage_increase = ((movies_2016_count - movies_2006_count) / movies_2006_count) * 100
print(f"Percentage increase in movies from 2006 to 2016: {percentage_increase:.2f}%")

## 10. Find the most common actor in all the movies.
all_actors = df['actors'].str.split(',').apply(lambda x: [actor.strip() for actor in x]).sum()
actor_counts = Counter(all_actors)
most_common_actor, most_common_count = actor_counts.most_common(1)[0]
print(f"Most common actor: {most_common_actor} with {most_common_count} appearances")

## 11. How many unique genres are there in the dataset?
unique_genres = set(df['genre'].str.split(',').sum())
print(f"Number of unique genres: {len(unique_genres)}")


## 12. Calculate the correlation matrix of the numerical features in the dataset
# Firstly, select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=[np.number])

# Next Calculate the correlation matrix of the numerical features in the dataset
correlation_matrix = numeric_df.corr()

# Firnally, Display the correlation matrix
print(correlation_matrix)


# THE END

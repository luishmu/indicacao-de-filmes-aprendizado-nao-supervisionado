#%% 

import pandas as pd 

links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

print(links.head())
print(movies.head())
print(ratings.head())
print(tags.head())
#%%

movies_name = pd.merge(movies,links, on='movieId', how='inner')
movies_name_rating = pd.merge(movies_name, ratings, on='movieId', how='inner')
movies_tags = pd.merge(movies, tags, on='movieId', how='inner')
movies_tags
#%%
ranking_films = movies_name_rating.groupby('title', as_index=False).agg({
    'rating': 'mean',
    'movieId': 'count'
})
ranking_films.rename(columns={'movieId': 'count'}, inplace=True)
ranking_films.sort_values(['count','rating'],ascending=[False, False], inplace=True)
ranking_films
# %%
tags_per_user = movies_tags.groupby('userId', as_index=False).agg({
    'tag': 'count'
})
tags_per_user.rename(columns={'tag': 'count_tags'}, inplace=True)
tags_per_user.sort_values(['count_tags'],ascending=[False], inplace=True)
tags_per_user.head(20)
# %%
movies_tags
# %%
# %%
# Quantidade de vezes que cada nota (0.5 a 5.0) foi dada
rating_distribution = ratings['rating'].value_counts().sort_index()
print(rating_distribution)
# %%
user_behavior = movies_name_rating.groupby('userId', as_index=False).agg({
    'rating': ['count', 'mean']
})
# Ajustando o nome das colunas após o agg múltiplo
user_behavior.columns = ['userId', 'quantidade_avaliacoes', 'media_notas']
user_behavior.sort_values(by='quantidade_avaliacoes', ascending=False, inplace=True)
user_behavior.head()
# %%
# Separa os gêneros numa lista e depois expande para linhas individuais
movies_exploded = movies.copy()
movies_exploded['genres'] = movies_exploded['genres'].str.split('|')
movies_exploded = movies_exploded.explode('genres')

# Conta quantos filmes existem por gênero
genres_count = movies_exploded['genres'].value_counts().reset_index()
genres_count.columns = ['genre', 'movie_count']
genres_count.head(10)


# %%
# Pegar apenas filmes com mais de 50 avaliações para ter relevância
filmes_relevantes = ranking_films[ranking_films['count'] > 50]['title']

polemicos = movies_name_rating[movies_name_rating['title'].isin(filmes_relevantes)]
polemicos = polemicos.groupby('title')['rating'].std().sort_values(ascending=False).reset_index()
polemicos.rename(columns={'rating': 'desvio_padrao'}, inplace=True)
print("Filmes que mais dividem opiniões:")
polemicos.head(10)

# %%
total_usuarios = ratings['userId'].nunique()
total_filmes = movies['movieId'].nunique()
total_avaliacoes = len(ratings)

possiveis_avaliacoes = total_usuarios * total_filmes
esparsidade = (1 - (total_avaliacoes / possiveis_avaliacoes)) * 100

print(f"A matriz é {esparsidade:.2f}% vazia.") 
# Isso mostra o quão difícil é o trabalho do algoritmo de recomendação!


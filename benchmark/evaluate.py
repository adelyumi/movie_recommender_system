import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from math import sqrt

# Define the model
class MovieRecommendationModel(nn.Module):
    def __init__(self, num_users, num_movies, num_features, embedding_dim=50, hidden_dim=100):
        super(MovieRecommendationModel, self).__init__()

        # Embedding layers for user and movie IDs
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # Fully connected layers for additional features
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 + num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user, movie, features):
        # Embed user and movie IDs
        user_embedding = self.user_embedding(user)
        movie_embedding = self.movie_embedding(movie)

        # Concatenate user and movie embeddings with additional features
        input_features = torch.cat([user_embedding, movie_embedding, features], dim=1)

        # Forward pass through fully connected layers
        output = self.fc(input_features)

        return output.squeeze()

def main():
    test = pd.read_csv('benchmark/data/test.csv')

    columns = list(test)
    columns = columns[5:]

    test_user_tensor = torch.LongTensor(test['user_id'].values)
    test_movie_tensor = torch.LongTensor(test['movie_id'].values)
    test_features_tensor = torch.FloatTensor(test[columns].values)
    test_ratings_tensor = torch.FloatTensor(test['rating'].values)

    # Load the saved best model state
    model = torch.load('models/model.pt')
    model.eval()

    # Make predictions on the test set
    with torch.no_grad():
        predictions_test = model(test_user_tensor, test_movie_tensor, test_features_tensor)
        loss_test = nn.MSELoss()(predictions_test, test_ratings_tensor)

    # Calculate RMSE for the test set
    rmse_test = sqrt(loss_test.item())
    print(f'RMSE on Test Set: {rmse_test:.4f}')

if __name__ == '__main__':
    main()

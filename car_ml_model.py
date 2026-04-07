import matplotlib.pyplot as plt
import pandas as pd
import os, joblib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from car_web_scrape import Brand_names_autotrader as brand_names

colors = ['#d62728', '#ff7f0e', '#ffdb58', '#2ca02c', '#1f77b4']  # red, orange, yellow, green, blue

model_datasets = {}
for name in brand_names:
    path1 = f'autotrader_{name}_dataset.csv'
    path2 = f'carzcoza_{name}_dataset.csv'

    if os.path.exists(path1):
        dataset = pd.read_csv(path1)
        print(f"Loaded: {path1}")
    elif os.path.exists(path2):
        dataset = pd.read_csv(path2)
        print(f"Loaded: {path2}")
    else:
        print(f"Warning: Neither {path1} nor {path2} found. Skipping {name}.")
        continue

    for value in dataset['Model'].unique():
        group_df = pd.DataFrame(dataset['Model'] == value)
        if len(group_df) > 1:
            model_datasets[value] = group_df


for name,ds in model_datasets.items:
    x = ds['Price_rating'].copy()
    for pr in x:
        if pr =='No rating':
            0
        elif pr =='High Price':
            1
        elif pr =='Low Price':
            2
        elif pr =='Fair Price':
            3
        elif pr =='Great Price':
            4

    y = ds['Price'].copy() 
    
    for rating in range(5):
        x_subset = (n for n in x == rating)
        plt.scatter(x_subset, 
                    y,
                    c=colors[rating],
                    label=f'Rating {rating}',
                    s=70,               # marker size
                    alpha=0.85,
                    edgecolors='black',
                    linewidth=0.6)

    # Formatting
    plt.xlabel('Price', fontsize=13, labelpad=10)
    plt.ylabel('Price Rating', fontsize=13, labelpad=10)
    plt.title('Scatter Plot: Price vs Price Rating', fontsize=15, pad=15)

    # Force y-axis to show only 0,1,2,3,4
    plt.yticks([0, 1, 2, 3, 4])

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Price Rating', title_fontsize=11, fontsize=10, loc='upper left')
    plt.show()
    
    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train,y_train)
    model.predict(X_test)
    model_filename = f'{name}_ml_model.joblib'
    joblib.dump(model,model_filename)



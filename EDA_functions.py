import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, n_cols: int = 3) -> plt.Figure:
    """
    Creates a grid of histograms for each feature in the dataset.

    Args:
        X (pd.DataFrame): DataFrame containing synthetic features.
        y (pd.Series): Series containing the target variable.
        n_cols (int): Number of columns in the grid layout.

    Returns:
        plt.Figure: The matplotlib Figure object containing the distribution plots.
    """
    features = X.columns
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            sns.histplot(X[feature], ax=ax, kde=True, color='skyblue')
            ax.set_title(f'Distribution of {feature}')
    
    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle('Feature Distributions', y=1.02, fontsize=16)
    plt.close(fig)
    return fig

def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series) -> plt.Figure:
    """
    Creates a correlation heatmap of all features and the target variable.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        y (pd.Series): Series containing the target variable.

    Returns:
        plt.Figure: The matplotlib Figure object containing the heatmap.
    """
    # Combine features and target into one DataFrame
    data = X.copy()
    data['target'] = y
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw the heatmap with a color bar
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap,
                center=0, square=True, linewidths=0.5, ax=ax)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16)
    plt.close(fig)
    return fig

def plot_feature_target_relationships(X: pd.DataFrame, y: pd.Series, n_cols: int = 3) -> plt.Figure:
    """
    Creates a grid of scatter plots showing the relationship between each feature and the target.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        y (pd.Series): Series containing the target variable.
        n_cols (int): Number of columns in the grid layout.

    Returns:
        plt.Figure: The matplotlib Figure object containing the relationship plots.
    """
    features = X.columns
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            # Scatter plot with regression line
            sns.regplot(x=X[feature], y=y, ax=ax, 
                       scatter_kws={'alpha': 0.5, 'color': 'blue'}, 
                       line_kws={'color': 'red'})
            ax.set_title(f'{feature} vs Target')
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle('Feature vs Target Relationships', y=1.02, fontsize=16)
    plt.close(fig)
    return fig

def plot_pairwise_relationships(X: pd.DataFrame, y: pd.Series, features: list[str]) -> plt.Figure:
    """
    Creates a pairplot showing relationships between selected features and the target.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        y (pd.Series): Series containing the target variable.
        features (List[str]): List of feature names to include in the plot.

    Returns:
        plt.Figure: The matplotlib Figure object containing the pairplot.
    """
    # Ensure features exist in the DataFrame
    valid_features = [f for f in features if f in X.columns]
    
    if not valid_features:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid features provided", ha='center', va='center')
        return fig
    
    # Combine selected features and target
    data = X[valid_features].copy()
    data['target'] = y
    
    # Create pairplot
    pairgrid = sns.pairplot(data, diag_kind="kde", 
                          plot_kws={"alpha": 0.6, "s": 50},
                          corner=True)
    
    pairgrid.fig.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=16)
    plt.close(pairgrid.fig)
    return pairgrid.fig

def plot_boxplots(X: pd.DataFrame, y: pd.Series, n_cols: int = 3) -> plt.Figure:
    """
    Creates a grid of box plots for each feature, with points colored by target value.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        y (pd.Series): Series containing the target variable.
        n_cols (int): Number of columns in the grid layout.

    Returns:
        plt.Figure: The matplotlib Figure object containing the box plots.
    """
    features = X.columns
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Create target bins for coloring
    y_binned = pd.qcut(y, 3, labels=['Low', 'Medium', 'High'])
    
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            # Box plot for each feature
            sns.boxplot(x=y_binned, y=X[feature], ax=ax)
            ax.set_title(f'Distribution of {feature} by Target Range')
            ax.set_xlabel('Target Range')
    
    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle('Feature Distributions by Target Range', y=1.02, fontsize=16)
    plt.close(fig)
    return fig

def plot_outliers(X: pd.DataFrame, n_cols: int = 3) -> plt.Figure:
    """
    Creates a grid of box plots to detect outliers in each feature.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        n_cols (int): Number of columns in the grid layout.

    Returns:
        plt.Figure: The matplotlib Figure object containing the outlier plots.
    """
    features = X.columns
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            # Box plot to detect outliers
            sns.boxplot(x=X[feature], ax=ax, color='skyblue')
            ax.set_title(f'Outlier Detection for {feature}')
            ax.set_xlabel(feature)
    
    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle('Outlier Detection for Features', y=1.02, fontsize=16)
    plt.close(fig)
    return fig

def plot_feature_importance(feature_importances: pd.Series) -> plt.Figure:
    """
    Creates a bar plot of feature importances.

    Args:
        feature_importances (pd.Series): Series containing feature names and their corresponding importance scores.

    Returns:
        plt.Figure: The matplotlib Figure object containing the feature importance plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort feature importances
    sorted_importances = feature_importances.sort_values(ascending=False)
    
    sns.barplot(x=sorted_importances.values, y=sorted_importances.index, ax=ax, palette='viridis')
    ax.set_title('Feature Importances', fontsize=16)
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()
    plt.close(fig)
    return fig


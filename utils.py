import warnings
warnings.filterwarnings('ignore', category=UserWarning)
# 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 
from scipy.stats import skew, kurtosis
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def get_trade_stats(rets, riskfree_rate=0.035):
    monthly_riskfree_rate = (1 + riskfree_rate)**(1/12) - 1

    annualized_return = rets.mean() * 12
    annualized_vol = rets.std() * np.sqrt(12)
    sharpe_ratio = (rets.mean() - monthly_riskfree_rate) / rets.std() * np.sqrt(12)

    downside_deviation = rets[rets < 0].std() * np.sqrt(12)
    cumulative_returns = (1 + rets).cumprod()
    peak = cumulative_returns.cummax()
    max_drawdown = ((peak - cumulative_returns) / peak).max()

    sortino_ratio = (rets.mean() - monthly_riskfree_rate) / downside_deviation

    return {
        "Annualized Return": annualized_return,
        "Annualized Vol": annualized_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Downside Deviation": downside_deviation,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Skewness": skew(rets.values),
        "Kurtosis": kurtosis(rets.values)
    }

def plot_log_rets(rets):
    fig, ax1 = plt.subplots(figsize=(18, 6))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Returns %', color='tab:red')
    ax1.plot(rets.index, rets['Overs'], label='Over Returns', color='orange')
    ax1.plot(rets.index, rets['Unders'], label='Under Returns', color='r')
    ax1.plot(rets.index, rets['Totals'], label='Total Returns', color='b')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.bar(rets.index, rets['Trade Count'], label='Trade Count', color='k', alpha=0.3, width=365//30)
    ax2.set_ylabel('Trades', color='k')
    ax2.set_ylabel('Trades', color='k')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')

    plt.title('Cumulative Returns and Trades')
    plt.show()

    return

def get_log_rets(re_overs_df, re_unders_df):
    """
    Calculates the logarithmic returns of over-performing and under-performing trades, along with their aggregate totals and trade counts.
    Parameters:
    - re_overs_df (pd.Series): A Series of returns for over-performing trades.
    - re_unders_df (pd.Series): A Series of returns for under-performing trades.
    Returns:
    - log_rets (pd.DataFrame): A DataFrame containing the logarithmic returns for overs, unders, their totals, and number of trades done.
    """
    o = re_overs_df.add(1)
    o = np.log(o / o.shift(1).fillna(0))
    u = re_unders_df.add(1)
    u = np.log(u / u.shift(1).fillna(0))

    log_rets = pd.concat([ o, u, o + u], axis=1)
    log_rets.columns = ['Overs', 'Unders', 'Totals']

    overs_count = overs_stock_df.groupby("DATE").size().resample('ME').sum()
    unders_count = unders_stock_df.groupby("DATE").size().resample('ME').sum()
    log_rets['Trade Count'] = overs_count.values + unders_count.values

    return log_rets

def plot_cummulative_rets(re_unders_df, re_overs_df):
    # Calculating cumulative returns
    re_unders_cum_df = re_unders_df.add(1).cumprod().sub(1).mul(100)
    re_overs_cum_df = re_overs_df.add(1).cumprod().sub(1).mul(100)
    total = (re_unders_cum_df + re_overs_cum_df) / 2
    peaks = total.cummax()

    # Creating a figure and axis objects
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # Creates 2 subplots vertically aligned

    # Plotting the first subplot
    axs[0].bar(re_unders_cum_df.index, re_unders_cum_df.values, label='Unders', color='b', alpha=0.4, width=365//4)
    axs[0].bar(re_overs_cum_df.index, re_overs_cum_df.values, label='Overs', color='r', alpha=0.4, width=365//4)
    axs[0].plot(re_overs_cum_df.index, total, label="Total", color='g')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Returns %')
    axs[0].legend()
    axs[0].set_title('Cumulative Returns Over Time')

    # Plotting the second subplot
    axs[1].plot(total.index, total.values, label="Total", color='g')
    axs[1].plot(peaks.index, peaks.values, label="Peak", color='k', linestyle='--')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Returns %')
    axs[1].legend()
    axs[1].set_title('Total Returns and Peaks Over Time')

    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
    plt.show()

    return

def get_pnl(trade_opportunities_df, MOM_FEATURES, pca_result_df, OVER_RETS, UNDER_RETS, fees=0.0015):
    """
    Calculates the profit and loss (PnL) for given trade opportunities, adjusting for fees, and aggregates the results on a monthly basis
    Parameters:
    - trade_opportunities_df (pd.DataFrame): A DataFrame containing trade opportunities with columns for date, overs, unders, and returns.
    - fees (float, optional): The trading fee rate to apply to the returns. Defaults to 0.0015.

    Returns:
    - overs_stock_df (pd.DataFrame): A DataFrame of over-performing trades merged with asset information, adjusted for fees.
    - unders_stock_df (pd.DataFrame): A DataFrame of under-performing trades merged with asset information, adjusted for fees.
    - overs_df (pd.DataFrame): The original over-performing trades DataFrame adjusted for fees and indexed by date.
    - unders_df (pd.DataFrame): The original under-performing trades DataFrame adjusted for fees and indexed by date.
    - re_unders_df (pd.Series): Monthly aggregated mean returns for under-performing trades.
    - re_overs_df (pd.Series): Monthly aggregated mean returns for over-performing trades.
    """
    trade_opportunities_df['DATE'] = pd.to_datetime(trade_opportunities_df['DATE'])
    asset_df = pca_result_df.reset_index()[['DATE', 'permno', MOM_FEATURES[0]]].copy()

    overs_df = trade_opportunities_df[["DATE", 'overs', OVER_RETS]].rename(columns={OVER_RETS: f'raw_{OVER_RETS}'})
    unders_df = trade_opportunities_df[["DATE", 'unders', UNDER_RETS]].rename(columns={UNDER_RETS: f'raw_{UNDER_RETS}'})
    overs_df[OVER_RETS] = overs_df[f'raw_{OVER_RETS}']
    unders_df[UNDER_RETS] = unders_df[f'raw_{UNDER_RETS}']

    overs_stock_df = pd.merge(overs_df.explode('overs').rename(columns={'overs': 'permno'}), asset_df, left_on=['DATE', 'permno'], right_on=['DATE', 'permno'])
    unders_stock_df = pd.merge(unders_df.explode('unders').rename(columns={'unders': 'permno'}), asset_df, left_on=['DATE', 'permno'], right_on=['DATE', 'permno'])

    # Add the trading fees
    overs_df[OVER_RETS] = overs_df[OVER_RETS].apply(lambda row: row- fees if row> 0 else row)
    unders_df[UNDER_RETS] = unders_df[UNDER_RETS].apply(lambda row: row- fees if row > 0 else row)
    unders_df = unders_df.set_index("DATE", drop=False).fillna(0)
    overs_df = overs_df.set_index("DATE", drop=False).fillna(0)

    # Resampling in case of multiple trades in one month
    re_unders_df = unders_df.resample('ME', on='DATE')[UNDER_RETS].mean()
    re_overs_df = overs_df.resample('ME', on='DATE')[OVER_RETS].mean()

    return overs_stock_df, unders_stock_df, overs_df, unders_df, re_unders_df, re_overs_df

def statarb_signals(group, MOM_FEATURES, RETS_1, std_dev_factor=1.5):
    """
    Finds signals for a given cluster of stocks by identifying overvalued ("overs") and undervalued ("unders") stocks
    in the top and bottom deciles, where the momentum difference exceeds a specified standard deviation factor, using pd.qcut.

    Parameters:
    - group (pd.DataFrame): Stock data for a specific group or cluster.
    - std_dev_factor (float, optional): The factor by which the standard deviation is multiplied to determine signal. Defaults to 1.

    Returns:
    - list of dicts:
        - 'DATE': The trade date for the identified signals.
        - 'Cluster': The identifier for the cluster from which the signals were generated.
        - 'overs': A set of stock identifiers ('permno') considered overvalued based on the strategy.
        - 'unders': A set of stock identifiers ('permno') considered undervalued based on the strategy.
    """
    overs = unders = []
    overs_rets = unders_rets = 0
    if len(group[MOM_FEATURES[0]]) > 1:
        group_sorted = group.sort_values(by=RETS_1, ascending=False).reset_index(drop=True)
        mid_idx = len(group_sorted) // 2
        top_half = group_sorted.iloc[:mid_idx]
        bottom_half = group_sorted.iloc[-mid_idx:]

        assert len(top_half) == len(bottom_half), f"len mismatch: {len(top_half) } != {len(bottom_half)}"

        if not bottom_half.empty and not top_half.empty:
            mom1_diffs = top_half[RETS_1].values - bottom_half[RETS_1].values
            mom1_std_dev = mom1_diffs.std()
            rets_diffs =  top_half[RETS_1].values - bottom_half[RETS_1].values
            valid_pairs_mask = rets_diffs > (mom1_std_dev * std_dev_factor)
            overs = top_half[valid_pairs_mask]['permno']
            unders = bottom_half[valid_pairs_mask]['permno']

            overs_rets = top_half[valid_pairs_mask][RETS].mul(-1).mean()
            unders_rets = bottom_half[valid_pairs_mask][RETS].mean()

    return [{
        'DATE': group['DATE_TRADE'].iloc[0],
        'Cluster': group['cluster'].iloc[0],
        'overs': set(overs),
        'unders': set(unders),
        OVER_RETS: overs_rets,
        UNDER_RETS: unders_rets,
        RETS: (overs_rets + unders_rets),
    }]


def process_trade_opportunities(df, cluster_label, filepath):
    df['cluster'] = df[cluster_label]

    tqdm.pandas(desc=f"StatArb opportunities with {cluster_label}")
    trade_opportunities = df.groupby(['cluster', 'DATE']).progress_apply(statarb_signals)
    trade_opportunities = [item for sublist in trade_opportunities for item in sublist]
    if len(trade_opportunities) == 0:
        return pd.DataFrame()

    trade_opportunities_df = pd.DataFrame(trade_opportunities)
    if not IS_KAGGLE:
        trade_opportunities_df.to_pickle(filepath)
    return trade_opportunities_df


def train_agg_clusters(pca_result_df, pca_components_cols, MOM_FEATURES, alpha=0.3):
    models_dfs = []

    cluster_membership = []
    for month, data in tqdm(pca_result_df.groupby("DATE"), desc="train_agg_clusters"):
        pca_data = data[pca_components_cols]
        if len(pca_data) < 2:
            print(f"Skipping {month} due to insufficient data.")
            continue

        _, eps = distance_to_nearest_neighbors(pca_data, alpha = alpha)
        agg_model = AgglomerativeClustering(n_clusters=None, distance_threshold=eps, linkage='average')
        agg_model.fit(pca_data)

        cluster_df = pd.DataFrame(data['permno'].copy(), index=data.index)
        cluster_df['agg_cluster'] = agg_model.labels_
        cluster_df['DATE'] = month
        cluster_df[MOM_FEATURES[0]] = data[MOM_FEATURES[0]]

        cluster_membership.append(cluster_df)

        models_dfs.append({'DATE': month, 'n_clusters': agg_model.n_clusters_})

    models_df = pd.DataFrame(models_dfs)
    cluster_membership_df = pd.concat(cluster_membership, ignore_index=False)

    return models_df, cluster_membership_df

def train_db_clusters(pca_result_df, pca_components_cols, MOM_FEATURES, alpha=0.1):
    models_dfs = []

    cluster_membership = []
    for month, data in tqdm(pca_result_df.groupby("DATE"), desc="train_db_clusters"):
        pca_data = data[pca_components_cols]
        if len(pca_data) < 2:
            print(f"Skipping {month} due to insufficient data.")
            continue

        #MinPts is set to be the natural logarithm of the total number of data points N
        min_samples = int(round(np.log(len(data))))
        _, eps = distance_to_nearest_neighbors(pca_data, k=min_samples + 1, alpha = alpha)
        db_model = DBSCAN(eps=eps, metric='l2', min_samples=min_samples)
        db_model.fit(pca_data)

        cluster_df = pd.DataFrame(data['permno'].copy(), index=data.index)
        cluster_df['db_cluster'] = db_model.labels_
        cluster_df['DATE'] = month
        cluster_df[MOM_FEATURES[0]] = data[MOM_FEATURES[0]]

        cluster_membership.append(cluster_df)
        # -1 are the noise points, which we have to remove.
        num_clusters = len(set(db_model.labels_)) - (1 if -1 in db_model.labels_ else 0)

        models_dfs.append({'DATE': month, 'n_clusters': num_clusters})

    models_df = pd.DataFrame(models_dfs)
    cluster_membership_df = pd.concat(cluster_membership, ignore_index=False)

    return models_df, cluster_membership_df

def distance_to_centroid(km_model, df, alpha=30):
    """
    Calculate the distance from each data point in the provided DataFrame to the centroid of its assigned cluster.
    Parameters:
    - km_model: Fitted KMeans model containing cluster centroids.
    - df: DataFrame containing the data points.
    - a: Optional, alpha max dist percentile to get epsilon threshold.

    Returns:
    - A tuple containing:
        - NumPy array of distances from each data point to its centroid.
        - Epsilon threshold, calculated as the alpha percentile of the L2 distances.
    """
    labels = km_model.labels_
    dist_centroid = km_model.transform(df)
    dist_centroid_member = np.array([dist_centroid[i, labels[i]] for i in range(len(labels))])
    epsilon = np.percentile(dist_centroid_member, alpha)

    return dist_centroid_member, epsilon

def plot_cluster_distributions(cluster_membership_df, cluster_col):
    """
    Plots the distribution of clusters over time, including a comparison of outliers versus clustered stocks.
    Parameters:
    - cluster_membership_df: DataFrame containing clustering information.
    - cluster_col: String, the name of the column in DataFrame that contains cluster identifiers.
    """
    clusters = cluster_membership_df[cluster_membership_df[cluster_col] > -1].groupby('DATE')[cluster_col].nunique().reset_index()
    clusters['DATE'] = pd.to_datetime(clusters['DATE'])
    cluster_counts = cluster_membership_df.groupby(['DATE', cluster_col])["permno"].count().reset_index()
    cluster_counts['Rank'] = cluster_counts[cluster_counts[cluster_col] > -1].groupby('DATE')["permno"].rank("dense", ascending=False)
    cluster_counts['Cluster_Group'] = cluster_counts.apply(
        lambda row: 'First' if row['Rank'] == 1 else
                    'Second' if row['Rank'] == 2 else
                    'Third' if row['Rank'] == 3 else
                    'Rest', axis=1)
    cluster_counts['Cluster_Group'] = cluster_counts.apply(
        lambda row: 'Outliers' if row[cluster_col] == -1 else
                    row['Cluster_Group'], axis=1)
    cluster_summary = cluster_counts.groupby(['DATE', 'Cluster_Group'])["permno"].sum().unstack(fill_value=0).reset_index()
    cluster_summary['DATE'] = pd.to_datetime(cluster_summary['DATE'])

    fig, axs = plt.subplots(1, 3 if 'Outliers' in cluster_summary.columns else 2, figsize=(18, 4))

    axs[0].fill_between(clusters['DATE'], clusters[cluster_col], step="pre", alpha=0.4)
    axs[0].set_title('Number of Clusters')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Number of Clusters')
    axs[0].tick_params(axis='x', rotation=45)
    axs[1].stackplot(cluster_summary['DATE'],
                     cluster_summary.get('First', pd.Series()),
                     cluster_summary.get('Second', pd.Series()),
                     cluster_summary.get('Third', pd.Series()),
                     cluster_summary.get('Rest', pd.Series()),
                     labels=['First', 'Second', 'Third', 'Rest'], alpha=0.4)
    axs[1].set_title('Stocks in 3 Largest Clusters')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Number of Stocks')
    axs[1].legend(loc='upper left')
    axs[1].tick_params(axis='x', rotation=45)

    if 'Outliers' in cluster_summary.columns:
        clustered_stocks = cluster_summary[['First', 'Second', 'Third', 'Rest']].sum(axis=1)
        outliers = cluster_summary['Outliers']
        axs[2].stackplot(cluster_summary['DATE'], outliers, clustered_stocks, labels=['Outliers', 'Clustered'], alpha=0.4)
        axs[2].set_title('Outliers vs Clustered')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Number of Stocks')
        axs[2].tick_params(axis='x', rotation=45)
        axs[2].legend()

    plt.tight_layout()
    plt.show()

    return

def train_km_clusters(pca_result_df, pca_components_cols, MOM_FEATURES, optimal_clusters = None):
    models_dfs = []
    cluster_membership = []
    for month, data in tqdm(pca_result_df.groupby("DATE"), desc="train_km_clusters"):
        pca_data = data[pca_components_cols]
        if len(pca_data) < 2:
            print(f"Skipping {month} due to insufficient data.")
            continue
        km_model = KMeans(n_clusters=optimal_clusters,
                          init='k-means++',
                          n_init=10,
                          max_iter=1000,
                          random_state=1)
        km_model.fit(pca_data)

        # We need to refit and remove the outlier securities.
        dist, eps = distance_to_centroid(km_model, pca_data)
        clean_data = pca_data[~(dist > eps)]
        cluster_df = pd.DataFrame(data['permno'].copy(), index=data.index)
        cluster_df = pd.DataFrame(pca_data['permno'].copy(), index=pca_data.index)
        cluster_df['km_cluster'] = km_model.labels_ if km_model is not None else []
        cluster_df.loc[(dist > eps), "km_cluster"] = -1 # Outliers To be similar to DBScan - we set them to -1
        cluster_df[MOM_FEATURES[0]] = data[MOM_FEATURES[0]]
        cluster_df['DATE'] = month
        cluster_membership.append(cluster_df)

        models_dfs.append({'DATE': month, 'n_clusters': km_model.n_clusters})

    models_df = pd.DataFrame(models_dfs)
    cluster_membership_df = pd.concat(cluster_membership, ignore_index=False)

    return models_df, cluster_membership_df

def interpolate_with_median(group, WINDOW):
    rolling_median = group.rolling(window=WINDOW, min_periods=1).median()
    group= group.fillna(rolling_median).bfill()
    return group

def distance_to_nearest_neighbors(df, k = 2, alpha = 0.3):
    """
    Calculates the distance to the nearest neighbors of each point in a DataFrame and determines a distance threshold based on a percentile.

    Computes the L2 nearest neighbors for each point in the dataset.
    It then calculates the average distance to the nearest neighbors, excluding the closest one (itself for k=2), across all points a threshold is used as cut-off distance for clustering or outlier detection.
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data points.
    - k (int, optional): The number of nearest neighbors to consider. The default is 2 (itself and another).
    - alpha (int, optional): The percentile from 1.0 to 0.0 to use when determining the distance threshold.

    Returns:
    - A tuple containing:
        - NumPy array of distances.
        - Epsilon threshold.
    """

    neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1, metric='l2')
    nbrs = neigh.fit(df)
    distances, _ = nbrs.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1:].mean(axis=1)
    epsilon = np.percentile(distances, alpha*100)

    return distances, epsilon
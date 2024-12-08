from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.simplefilter("error", RuntimeWarning)

from pathlib import Path
import os
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import distributions, kstest, chi2_contingency
from matplotlib.ticker import MultipleLocator, MaxNLocator

from .utils import transform_params, get_dist_name, get_representitives, vote_representitives

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams["axes.unicode_minus"] = False  # Use ASCII hyphen for minus


def plot_distribution_comparison(save_dir: Path, weekday_df: pd.DataFrame, weekend_df: pd.DataFrame, weekday_distribution_df: pd.DataFrame, weekend_distribution_df: pd.DataFrame, station_id: int, station_name: str, time_column: str, on: bool, debug: bool = False):
    weekday_data = weekday_df[weekday_df['역번호'] == station_id][time_column]
    weekend_data = weekend_df[weekend_df['역번호'] == station_id][time_column]

    weekday_Q1, weekday_Q3 = weekday_data.quantile(0.25), weekday_data.quantile(0.75)
    weekday_IQR = weekday_Q3 - weekday_Q1
    lower_bound = weekday_Q1 - weekday_IQR
    upper_bound = weekday_Q3 + weekday_IQR
    filtered_weekday_data = weekday_data[(weekday_data >= lower_bound) & (weekday_data <= upper_bound)]
    if debug:
        print(f"Weekday original data size: {len(weekday_data)}")
        print(f"Weekday filtered data size: {len(filtered_weekday_data)}")

    weekend_Q1, weekend_Q3 = weekend_data.quantile(0.25), weekend_data.quantile(0.75)
    weekend_IQR = weekend_Q3 - weekend_Q1
    lower_bound = weekend_Q1 - weekend_IQR
    upper_bound = weekend_Q3 + weekend_IQR
    filtered_weekend_data = weekend_data[(weekend_data >= lower_bound) & (weekend_data <= upper_bound)]
    if debug:
        print(f"Weekday original data size: {len(weekend_data)}")
        print(f"Weekday filtered data size: {len(filtered_weekend_data)}")

    weekday_data_size = weekday_distribution_df[(weekday_distribution_df['역번호'] == station_id) & (weekday_distribution_df['시간대'] == time_column)]['데이터수'].values[0]
    weekend_data_size = weekend_distribution_df[(weekend_distribution_df['역번호'] == station_id) & (weekend_distribution_df['시간대'] == time_column)]['데이터수'].values[0]
    assert len(filtered_weekday_data) == weekday_data_size
    assert len(filtered_weekend_data) == weekend_data_size

    weekday_dist = weekday_distribution_df[(weekday_distribution_df['역번호'] == station_id) & (weekday_distribution_df['시간대'] == time_column)]['분포'].values[0]
    weekend_dist = weekend_distribution_df[(weekend_distribution_df['역번호'] == station_id) & (weekend_distribution_df['시간대'] == time_column)]['분포'].values[0]

    weekday_dist_name = get_dist_name(weekday_dist)
    weekend_dist_name = get_dist_name(weekend_dist)

    weekday_dist_params = weekday_distribution_df[(weekday_distribution_df['역번호'] == station_id) & (weekday_distribution_df['시간대'] == time_column)]['매개변수'].values[0]
    weekend_dist_params = weekend_distribution_df[(weekend_distribution_df['역번호'] == station_id) & (weekend_distribution_df['시간대'] == time_column)]['매개변수'].values[0]

    weekday_representitives = get_representitives(filtered_weekday_data)
    weekend_representitives = get_representitives(filtered_weekend_data)

    vote_result, vote_won = vote_representitives('평일', '주말', weekday_representitives, weekend_representitives)

    combined_data = pd.concat([filtered_weekday_data, filtered_weekend_data], axis=0)
    Q1, Q3 = combined_data.quantile([0.25, 0.75])
    IQR = Q3 - Q1

    bin_width = IQR / np.cbrt(len(combined_data)) * .5
    if bin_width == 0:
        bin_width = 20
    if debug:
        print(f"Bin Width (Freedman-Diaconis, Combined): {bin_width}")

    min_val = min(filtered_weekday_data.min(), filtered_weekend_data.min())
    max_val = max(filtered_weekday_data.max(), filtered_weekend_data.max())
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    if debug:
        print("min_val:", min_val)
        print("max_val:", max_val)
        print("bin_width:", bin_width)

    weekday_x = np.linspace(filtered_weekday_data.min(), filtered_weekday_data.max(), 1000)
    weekend_x = np.linspace(filtered_weekend_data.min(), filtered_weekend_data.max(), 1000)

    weekday_pdf = getattr(distributions, weekday_dist_name).pdf(weekday_x, *weekday_dist_params[:-2], loc=weekday_dist_params[-2], scale=weekday_dist_params[-1]) * len(filtered_weekday_data) * bin_width
    weekend_pdf = getattr(distributions, weekend_dist_name).pdf(weekend_x, *weekend_dist_params[:-2], loc=weekend_dist_params[-2], scale=weekend_dist_params[-1]) * len(filtered_weekend_data) * bin_width

    plt.figure(figsize=(10, 6))

    plt.hist(filtered_weekday_data, bins=bins, alpha=0.5, label=f'평일 (n={weekday_data_size})', edgecolor='black', color='blue')
    plt.plot(weekday_x, weekday_pdf, color='blue', linestyle='--', label=f"Fit: {weekday_dist}")

    plt.hist(filtered_weekend_data, bins=bins, alpha=0.5, label=f'주말 (n={weekend_data_size})', edgecolor='black', color='red')
    plt.plot(weekend_x, weekend_pdf, color='red', linestyle='--', label=f"Fit: {weekend_dist}")

    plt.text(
        0.05, 0.75,
        f"{vote_won} > {'평일' if vote_won == '주말' else '주말'} ({','.join(vote_result[vote_won])})",
        transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8)
    )

    plt.grid(alpha=0.4)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
    plt.title(f"역번호: {station_id}, 역명: {station_name}\n구분: 승차, 시간대: {time_column}", fontsize=14, fontweight='bold')
    plt.xlabel("인원수", fontsize=12)
    plt.ylabel("빈도수", fontsize=12)
    plt.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    save_title = f"{station_id}_{station_name}_{'승차' if on else '하차'}_{time_column}_평일_주말_분포_비교"
    plt.savefig(save_dir / f"{save_title}.png", dpi=200)
    if debug:
        plt.show()
    plt.close('all')

    return {
        "역번호": station_id,
        "역명": station_name,
        "구분": "승차" if on else "하차",
        "시간대": time_column,
        "평일 분포": weekday_dist,
        "평일 매개변수": weekday_dist_params,
        "평일 데이터수": weekday_data_size,
        "주말 분포": weekend_dist,
        "주말 매개변수": weekend_dist_params,
        "주말 데이터수": weekend_data_size,
        "승자": vote_won,
        "승리 횟수": len(vote_result[vote_won]),
        "승리 항목": vote_result[vote_won],
    }


def plot_timeline_ratio(save_dir: Path, subway_df: pd.DataFrame, bus_df: pd.DataFrame, time_columns: list, station_id: str, station_name: str, category: str, debug: bool):
    subway_origin_row = subway_df[(subway_df['역번호'] == station_id) & (subway_df['구분'] == category)]
    bus_origin_row = bus_df[(bus_df['역번호'] == station_id) & (bus_df['구분'] == category)]

    assert len(subway_origin_row) == 1
    assert len(bus_origin_row) == 1

    contingency_table = pd.concat([subway_origin_row[time_columns], bus_origin_row[time_columns]], axis=0).values
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    if debug:
        print(f"Chi-squared statistic: {chi2}")
        print(f"P-value: {p}")
        print(f"Same: {p > 0.05}")
        print(f"Degrees of freedom: {dof}")
        print(f"Expected frequencies for Subway:\n{expected[0]}")
        print(f"Expected frequencies for Bus:\n{expected[1]}")
    

    plt.figure(figsize=(8, 6))

    plt.plot(time_columns, subway_origin_row[time_columns].values[0], marker='o', label='Subway', color='r', alpha=0.8, linewidth=2)
    plt.plot(time_columns, bus_origin_row[time_columns].values[0], marker='o', label='Bus', color='b', alpha=0.8, linewidth=2)
    plt.plot(time_columns, expected[0], marker='*', linestyle='--', label='Expected (Subway)', color='y', alpha=0.8, linewidth=1)
    plt.plot(time_columns, expected[1], marker='*', linestyle='--', label='Expected (Bus)', color='c', alpha=0.8, linewidth=1)


    plt.grid(alpha=0.4)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
    # plt.gca().yaxis.set_major_locator(MultipleLocator(.05))
    plt.xticks(rotation=45)

    plt.title(f'역번호: {station_id}, 역명: {station_name}, 구분: {category}', fontsize=14, fontweight='bold')
    plt.xlabel('시간대')
    plt.ylabel('비율')
    plt.legend(loc='upper right', fontsize=10)

    plt.text(0.05, 0.7, f'Chi-sq: {chi2:.2f}\nP-value: {p:.2f}\nSame: {p > 0.05}', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'), 
            transform=plt.gca().transAxes)

    plt.tight_layout()
    save_title = f"{station_id}_{station_name}_{category}_시간대별_비율_분포_비교"
    plt.savefig(save_dir / f"{save_title}.png", dpi=200)
    if debug:
        plt.show()
    plt.close('all')

    return {
        '역번호': station_id,
        '역명': station_name,
        '구분': category,
        '카이제곱통계량': chi2,
        'P값': p,
        '자유도': dof,
        '지하철 기대 빈도': expected[0],
        '버스 기대 빈도': expected[1],    
        '동일 여부': '동일' if p > 0.05 else '다름',
    }


def plot_distribution(save_dir, df, on, station_id, station_name, timeline, candidate_distributions, debug=False):
    assert station_id in df['역번호'].unique(), f"Station ID {station_id} not found in the DataFrame"
    assert df[df['역번호'] == station_id]['역명'].nunique() == 1, f"Multiple station names found for ID {station_id}"
    assert df[df['역번호'] == station_id]['역명'].unique().tolist()[0] == station_name, f"Station name mismatch for ID {station_id}"
    assert timeline in df.columns, f"Timeline {timeline} not found in the DataFrame"
    
    # Cutoff outliers using IQR
    data = df[df['역번호'] == station_id][timeline]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR
    upper_bound = Q3 + IQR
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    if debug:
        print(f"Original data size: {len(data)}")
        print(f"Filtered data size: {len(filtered_data)}")

    filtered_data = filtered_data.values

    # Store results
    results = []

    for dist_name in candidate_distributions:
        if station_id == 224 and timeline == '07시-08시' and on: # 맞추기는 성공했는데, 너무 말도 안되는 분포로 맞춰진 경우 삭제
            break
        if station_id == 203 and timeline == '24시-01시' and not on:
            break
        try:
            # Get the distribution
            dist = getattr(distributions, dist_name)

            try:
                # Fit the distribution to the data
                params = dist.fit(filtered_data)

                # Perform the Kolmogorov-Smirnov test
                ks_stat, p_value = kstest(filtered_data, dist_name, args=params)
            except RuntimeWarning as e:
                continue

            if p_value < 0.05: # reject the null hypothesis
                continue

            # Append results
            results.append({
                "Distribution": dist_name,
                "KS Statistic": ks_stat,
                "P-Value": p_value,
                "Parameters": params,
                "Success": "성공"
            })

        except Exception as e:
            # Skip distributions that fail
            print(f"Could not fit {dist_name}: {e}")
    
    if len(results) == 0:
        if debug:
            print("No distributions were fit successfully, Try arbitrary triangular distribution")
        dist_name = 'triang'
        dist = getattr(distributions, dist_name)
        try:
            params = dist.fit(filtered_data)
            if params[1] < 0: # loc must be non-negative
                params = dist.fit(filtered_data, floc=0)
            ks_stat, p_value = kstest(filtered_data, dist_name, args=params)
        except RuntimeWarning as e:
            pass
        results.append({
            "Distribution": dist_name,
            "KS Statistic": ks_stat,
            "P-Value": p_value,
            "Parameters": params,
            "Success": "실패"
        })

    results_df = pd.DataFrame(results).sort_values(by="KS Statistic")
    if debug:
        print(results_df)

    # Plot the best distribution
    best_fit = results_df.iloc[0]
    best_dist_name = best_fit["Distribution"]
    best_ks_stat = best_fit["KS Statistic"]
    best_p_value = best_fit["P-Value"]
    best_params = best_fit["Parameters"]
    best_params_transformed = transform_params(best_dist_name, best_params)
    best_success = best_fit["Success"]

    # Get the best-fitting distribution
    best_dist = getattr(distributions, best_dist_name)

    # Plot histogram of data
    plt.figure(figsize=(8, 6))
    counts, bins, _ = plt.hist(filtered_data, bins=30, density=False, alpha=0.6, color='blue', label='Data Histogram')
    bin_width = bins[1] - bins[0]

    # Generate x values for the PDF
    x = np.linspace(min(filtered_data), max(filtered_data), 1000)
    pdf = best_dist.pdf(x, *best_params[:-2], loc=best_params[-2], scale=best_params[-1]) * len(filtered_data) * bin_width # Compute the PDF

    # Plot best-fit PDF
    plt.plot(x, pdf, 'r-', lw=2, label=f"Best Fit: {best_params_transformed}")

    # Add K-S statistic and p-value to the plot
    plt.text(
        0.05, 0.75,
        f"K-S Statistic: {best_ks_stat:.4f}\nP-Value: {best_p_value:.4f}",
        transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8)
    )

    # Add grid, legend, and labels
    plt.grid(alpha=0.4)
    plt.gca().yaxis.set_major_locator(MultipleLocator(2))
    # plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.title(f"역번호: {station_id}, 역명: {station_name}\n구분: {'승차' if on else '하차'}, 시간대: {timeline} (n={len(filtered_data)}일){' (임의)' if best_success == '실패' else ''}", fontsize=14, fontweight='bold')
    plt.xlabel("인원수", fontsize=12)
    plt.ylabel("빈도수", fontsize=12)
    plt.legend(loc='upper left', fontsize=10)

    # Show plot
    plt.tight_layout()
    save_title = f"{station_id}_{station_name}_{'승차' if on else '하차'}_{timeline}"
    plt.savefig(save_dir / f"{save_title}.png", dpi=200)
    if debug:
        plt.show()
    plt.close('all')

    return {
        "역번호": station_id,
        "역명": station_name,
        "구분": "승차" if on else "하차",
        "시간대": timeline,
        "분포": best_params_transformed,
        "매개변수": best_params,
        "KS통계량": round(best_ks_stat.item(), 6),
        "P값": round(best_p_value.item(), 6),
        "데이터수": len(filtered_data),
        "성공여부": best_success 
    }

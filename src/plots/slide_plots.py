import pandas as pd
from src.config import pathData, pathImages
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np
import matplotlib.pyplot as plt

PATH_IMAGES_OTHERS = f'{pathImages}/others/'

def llm_release_plot():
    """
    This function reads an Excel file and displays its content.
    The file should contain data related to LLM releases.
    """
        
    # Path to the Excel file
    file_path = pathData + '2025 LifeArchitect.ai.xlsx'

    # Open the Excel file
    df = pd.read_excel(file_path)
    key = 'Announced\n▼'

    # Filter out rows where "Announced" is None
    df = df[df[key].notna()]
    # Convert the 'Announced' column to datetime format
    df = df[df[key] != 'TBA']
    df[key] = pd.to_datetime(df[key], format='%b/%Y')

    # Keep only model and key columns
    df = df[['Model', key]]
    
    # Group by month and count occurrences
    monthly_counts = df.groupby(df[key].dt.to_period('M')).size()
    # Convert period index back to datetime for plotting
    monthly_counts.index = monthly_counts.index.to_timestamp()

    # Filter data to start from 2021-01
    start_date = pd.to_datetime('2021-01-01')
    monthly_counts = monthly_counts[monthly_counts.index >= start_date]

    # Filter data to end at 2025-12
    end_date = pd.to_datetime('2025-04-30')
    monthly_counts = monthly_counts[monthly_counts.index <= end_date]
    
    # Create a line plot
    plt.figure(figsize=(12, 6))

    plt.plot(monthly_counts.index, monthly_counts.values, 'o', color='lightblue', alpha=0.8, label='Data Points')
    plt.plot(monthly_counts.index, monthly_counts.values, '-', color='blue', alpha=0.6)

    # Set x-axis formatting
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(MonthLocator(bymonth=[6, 12]))  # Show ticks every 6 months

    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    # Set labels for the axes
    plt.title('Number of LLM Releases Over Time', fontsize=18)
    plt.xlabel('Month/Year', fontsize=16)
    plt.ylabel('Number of Announced Models', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    # Display the plot
    plt.savefig(PATH_IMAGES_OTHERS + 'llm_releases_over_time.pdf', bbox_inches='tight')

def llm_release_plot_trendline():# Path to the Excel file
    file_path = pathData + '2025 LifeArchitect.ai.xlsx'

    # Open the Excel file
    df = pd.read_excel(file_path)
    key = 'Announced\n▼'

    # Filter out rows where "Announced" is None
    df = df[df[key].notna()]
    # Convert the 'Announced' column to datetime format
    df = df[df[key] != 'TBA']
    df[key] = pd.to_datetime(df[key], format='%b/%Y')

    # Keep only model and key columns
    df = df[['Model', key]]
    
    # Group by month and count occurrences
    monthly_counts = df.groupby(df[key].dt.to_period('M')).size()
    # Convert period index back to datetime for plotting
    monthly_counts.index = monthly_counts.index.to_timestamp()

    # Filter data to start from 2021-01
    start_date = pd.to_datetime('2021-01-01')
    monthly_counts = monthly_counts[monthly_counts.index >= start_date]

    # Filter data to end at 2025-12
    end_date = pd.to_datetime('2025-04-30')
    monthly_counts = monthly_counts[monthly_counts.index <= end_date]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot original data points
    plt.plot(monthly_counts.index, monthly_counts.values, 'x', color='black', alpha=1.0, markersize=1, label='Data Points')
    
    # Create smooth trend line using polynomial fit
    x_numeric = np.arange(len(monthly_counts))
    coeffs = np.polyfit(x_numeric, monthly_counts.values, deg=3)  # 3rd degree polynomial
    trend_line = np.polyval(coeffs, x_numeric)
    
    plt.plot(monthly_counts.index, trend_line, '-', color='blue', linewidth=2, label='Trend Line')
    plt.legend(fontsize=16)
    # Set x-axis formatting
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(MonthLocator(bymonth=[6, 12]))  # Show ticks every 6 months

    plt.yticks(fontsize=14)
    plt.xticks(rotation=45, fontsize=14)
    # Set labels for the axes
    plt.title('Number of LLM Releases Over Time', fontsize=18)
    plt.xlabel('Month/Year', fontsize=16)
    plt.ylabel('Number of Announced Models', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Display the plot
    plt.savefig(PATH_IMAGES_OTHERS + 'llm_releases_trendline.pdf', bbox_inches='tight')


def llm_release_plot_area():# Path to the Excel file
    file_path = pathData + '2025 LifeArchitect.ai.xlsx'

    # Open the Excel file
    df = pd.read_excel(file_path)
    key = 'Announced\n▼'

    # Filter out rows where "Announced" is None
    df = df[df[key].notna()]
    # Convert the 'Announced' column to datetime format
    df = df[df[key] != 'TBA']
    df[key] = pd.to_datetime(df[key], format='%b/%Y')

    # Keep only model and key columns
    df = df[['Model', key]]
    
    # Group by month and count occurrences
    monthly_counts = df.groupby(df[key].dt.to_period('M')).size()
    # Convert period index back to datetime for plotting
    monthly_counts.index = monthly_counts.index.to_timestamp()

    # Filter data to start from 2021-01
    start_date = pd.to_datetime('2021-01-01')
    monthly_counts = monthly_counts[monthly_counts.index >= start_date]

    # Filter data to end at 2025-12
    end_date = pd.to_datetime('2025-04-30')
    monthly_counts = monthly_counts[monthly_counts.index <= end_date]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot area under the curve
    plt.fill_between(monthly_counts.index, monthly_counts.values, color='lightblue', alpha=0.6, label='Area Under Curve')
    plt.plot(monthly_counts.index, monthly_counts.values, '-', color='darkblue', alpha=0.6, linewidth=0.7, label='Outer Line')
    # Remove the legend
    plt.legend().remove()

    
    # Set x-axis formatting
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(MonthLocator(bymonth=[6, 12]))  # Show ticks every 6 months

    plt.yticks(fontsize=14)
    plt.xticks(rotation=45, fontsize=14)
    # Set labels for the axes
    plt.title('Number of LLM Releases Over Time', fontsize=18)
    plt.xlabel('Month/Year', fontsize=16)
    plt.ylabel('Number of Announced Models', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Display the plot
    plt.savefig(PATH_IMAGES_OTHERS + 'llm_releases_area.pdf', bbox_inches='tight')

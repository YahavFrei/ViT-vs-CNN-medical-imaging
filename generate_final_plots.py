import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

def plot_comparison_bar_charts():
    """
    Generates three key visualization plots based on the ACTUAL results from 'final_run_output.out'.
    Corrected specifically for ResNet SGD (Final Log Entry).
    """
    
    # Ensure the output directory exists
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # DATA PREPARATION (Corrected Final Values)
    # ==========================================
    data = {
        'Model': ['DenseNet121', 'ResNet50', 'ViT Base', 'ViT (No Aug)', 'ResNet (SGD)'],
        
        # Updated 'ResNet (SGD)' to 83.81 (Run 7) instead of 87.34 (Run 6)
        'Accuracy': [93.91, 91.83, 88.94, 79.17, 83.81],
        
        # Updated 'ResNet (SGD)' to 98.21
        'Recall':    [98.97, 96.67, 99.49, 100.00, 98.21],
        
        # Updated 'ResNet (SGD)' to 80.29
        'Precision': [91.90, 90.84, 85.27, 75.00, 80.29]
    }
    df = pd.DataFrame(data)

    # ==========================================
    # PLOT 1: Top 3 Models Comparison
    # ==========================================
    print("Generating Plot 1: Top 3 Models Comparison...")
    plt.figure(figsize=(10, 6))
    
    # Only comparing the optimized versions (Top 3)
    best_models = df[df['Model'].isin(['DenseNet121', 'ResNet50', 'ViT Base'])]
    
    melted = best_models.melt(id_vars='Model', value_vars=['Accuracy', 'Recall', 'Precision'], var_name='Metric', value_name='Score')
    
    sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.ylim(80, 100)
    plt.title('Top 3 Models Comparison: DenseNet Wins on Accuracy', fontsize=14)
    plt.ylabel('Score (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'final_comparison_top3.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

    # ==========================================
    # PLOT 2: Impact of Augmentation
    # ==========================================
    print("Generating Plot 2: Impact of Augmentation...")
    plt.figure(figsize=(8, 6))
    
    aug_data = pd.DataFrame({
        'Architecture': ['DenseNet121', 'DenseNet121', 'ViT Base', 'ViT Base'],
        'Setup': ['Optimized (With Aug)', 'No Augmentation', 'Optimized (With Aug)', 'No Augmentation'],
        # Using the exact numbers from previous check for No-Aug runs
        'Accuracy': [93.91, 74.68, 88.94, 79.17]
    })
    
    sns.barplot(data=aug_data, x='Architecture', y='Accuracy', hue='Setup', palette='rocket')
    plt.ylim(60, 100)
    plt.title('Impact of Data Augmentation: ViT vs. CNN', fontsize=14)
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'impact_of_augmentation.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

    # ==========================================
    # PLOT 3: Optimizer Trade-off (ResNet Specific)
    # ==========================================
    print("Generating Plot 3: Optimizer Trade-off...")
    plt.figure(figsize=(8, 6))
    
    # Comparing ResNet AdamW (91.83) vs ResNet SGD (83.81)
    optimizer_data = pd.DataFrame({
        'Optimizer': ['AdamW (Optimized)', 'SGD'],
        # SGD Recall updated to 98.21, Precision to 80.29
        'Recall': [96.67, 98.21],
        'Precision': [90.84, 80.29]
    })
    melted_opt = optimizer_data.melt(id_vars='Optimizer', var_name='Metric', value_name='Score')
    
    sns.barplot(data=melted_opt, x='Optimizer', y='Score', hue='Metric', palette='coolwarm')
    plt.ylim(70, 100) # Adjusted ylim to catch the lower precision of SGD
    plt.title('ResNet50: AdamW vs. SGD (Sensitivity Trade-off)', fontsize=14)
    plt.ylabel('Score (%)')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'optimizer_tradeoff.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_comparison_bar_charts()

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import pandas as pd
# import os

# def plot_comparison_bar_charts():
#     """
#     Generates three key visualization plots based on the final experiment results.
#     Saves the plots to the 'results/plots/' directory.
#     """
    
#     # Ensure the output directory exists
#     output_dir = 'results/plots'
#     os.makedirs(output_dir, exist_ok=True)

#     # ==========================================
#     # DATA PREPARATION (Based on Final Log Analysis)
#     # ==========================================
#     # We use hardcoded values here to ensure the plots exactly match 
#     # the reported table in the document.
#     data = {
#         'Model': ['DenseNet121', 'ResNet50', 'ViT Base', 'ViT (No Aug)', 'ResNet (SGD)'],
#         'Accuracy': [93.11, 89.42, 88.46, 82.53, 87.50],
#         'Recall': [96.67, 97.69, 96.67, 95.90, 98.46],
#         'Precision': [92.77, 87.02, 86.61, 80.65, 84.38]
#     }
#     df = pd.DataFrame(data)

#     # ==========================================
#     # PLOT 1: Top 3 Models Comparison (The "Winner" Chart)
#     # ==========================================
#     print("Generating Plot 1: Top 3 Models Comparison...")
#     plt.figure(figsize=(10, 6))
    
#     # Filter only the optimized versions of the 3 models
#     best_models = df[df['Model'].isin(['DenseNet121', 'ResNet50', 'ViT Base'])]
    
#     # Melt dataframe for Seaborn plotting
#     melted = best_models.melt(id_vars='Model', value_vars=['Accuracy', 'Recall', 'Precision'], var_name='Metric', value_name='Score')
    
#     sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette='viridis')
#     plt.ylim(80, 100) # Zoom in on the relevant range
#     plt.title('Top 3 Models Comparison: DenseNet Wins on Accuracy', fontsize=14)
#     plt.ylabel('Score (%)')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
    
#     save_path = os.path.join(output_dir, 'final_comparison_top3.png')
#     plt.savefig(save_path)
#     print(f"Saved: {save_path}")

#     # ==========================================
#     # PLOT 2: Impact of Augmentation (The "ViT Weakness" Chart)
#     # ==========================================
#     print("Generating Plot 2: Impact of Augmentation...")
#     plt.figure(figsize=(8, 6))
    
#     aug_data = pd.DataFrame({
#         'Architecture': ['DenseNet121', 'DenseNet121', 'ViT Base', 'ViT Base'],
#         'Setup': ['Optimized (With Aug)', 'No Augmentation', 'Optimized (With Aug)', 'No Augmentation'],
#         'Accuracy': [93.11, 89.10, 88.46, 82.53]
#     })
    
#     sns.barplot(data=aug_data, x='Architecture', y='Accuracy', hue='Setup', palette='rocket')
#     plt.ylim(75, 95)
#     plt.title('Impact of Data Augmentation: ViT vs. CNN', fontsize=14)
#     plt.ylabel('Accuracy (%)')
#     plt.tight_layout()
    
#     save_path = os.path.join(output_dir, 'impact_of_augmentation.png')
#     plt.savefig(save_path)
#     print(f"Saved: {save_path}")

#     # ==========================================
#     # PLOT 3: Optimizer Trade-off (ResNet Specific)
#     # ==========================================
#     print("Generating Plot 3: Optimizer Trade-off...")
#     plt.figure(figsize=(8, 6))
    
#     optimizer_data = pd.DataFrame({
#         'Optimizer': ['AdamW (Optimized)', 'SGD'],
#         'Recall': [97.69, 98.46],
#         'Precision': [87.02, 84.38]
#     })
#     melted_opt = optimizer_data.melt(id_vars='Optimizer', var_name='Metric', value_name='Score')
    
#     sns.barplot(data=melted_opt, x='Optimizer', y='Score', hue='Metric', palette='coolwarm')
#     plt.ylim(80, 100)
#     plt.title('ResNet50: AdamW vs. SGD (Sensitivity Trade-off)', fontsize=14)
#     plt.ylabel('Score (%)')
#     plt.tight_layout()
    
#     save_path = os.path.join(output_dir, 'optimizer_tradeoff.png')
#     plt.savefig(save_path)
#     print(f"Saved: {save_path}")

# if __name__ == "__main__":
#     plot_comparison_bar_charts()
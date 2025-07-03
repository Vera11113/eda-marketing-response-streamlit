import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from matplotlib.cbook import boxplot_stats
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap


sns.set_theme(
    style='whitegrid',
    font_scale=1.2
)
cmap = ListedColormap(['skyblue', 'salmon'])

plt.rcParams.update({
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "axes.labelsize":   9,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "axes.prop_cycle":  plt.cycler("color", ["skyblue"]),
    "figure.constrained_layout.use": True,
})

def plot_boxplot(col, data):
    fig = plt.figure(figsize=(6, 4), dpi = 500)
    sns.boxplot(data[col], orient='h')
    plt.title(f'Boxplot: {col}')
    fig.tight_layout()
    fig.savefig("boxplot_temp.png", bbox_inches="tight")  # сохраняем
    st.image("boxplot_temp.png", width=500)  # выводим с нужной шириной

def plot_hist(col, data):
    fig = plt.figure(figsize=(6, 4), dpi = 500)
    sns.histplot(data[col], kde=True)
    plt.title(f'Histplot: {col}')
    fig.tight_layout()
    fig.savefig("histplot.png", bbox_inches="tight")  # сохраняем
    st.image("histplot.png", width=500)

def plot_log_hist(col, data):
    fig = plt.figure(figsize=(6, 4), dpi = 500)
    sns.histplot(np.log1p(data[col]), kde=True)
    plt.title(f'Log Histplot: {col}')
    fig.tight_layout()
    fig.savefig("Log Histplot.png", bbox_inches="tight")  # сохраняем
    st.image("Log Histplot.png", width=500)

def plot_corr_num(num_cols, data):
    fig = plt.figure(figsize=(6, 4), dpi=500)
    corr = data[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    fig.savefig("correlation_heatmap_num_features.png", bbox_inches="tight")  # сохраняем
    st.image("correlation_heatmap_num_features.png", use_container_width=True)

def plot_barplot(col, data):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)

    order = data[col].value_counts().index.to_list()
    sns.countplot(data=data, x=col, hue='TARGET', order=order, ax=ax, palette=['skyblue', 'salmon'])

    ax.set_title(f'Кол-во положительных и негативных откликов в группах {col}', fontsize=14)
    ax.set_ylabel('Count', fontsize = 12)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=10, rotation = 90)

    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontsize=10)

    ax.legend(
        title='TARGET',
        loc='upper right',
        fontsize=8,
        title_fontsize=10
    )

    filename = f'Barplot_{col}.png'
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    st.image(filename, width=500)

def plot_positive_barplot_in_group(col, data):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    order = data[col].value_counts().index.to_list()
    df_rate = data.groupby(col)['TARGET'].mean().mul(100).reset_index(name='response_rate')
    sns.barplot(data=df_rate, x=col, y='response_rate', order=order, color='skyblue', ax=ax)
    ax.set_title(f'Доля положительного отклика по группам {col}', fontsize=14)
    ax.set_ylabel('Count', fontsize = 12)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=10, rotation = 90)
    for c in ax.containers:
       ax.bar_label(c, fmt='%.1f%%', fontsize=10)
    filename = f'positive_barplot_in_group_{col}.png'
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    st.image(filename, width=500)

def plot_positive_barplot(col, data):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    order = data[col].value_counts().index.to_list()
    df_share = data.loc[data['TARGET'] == 1, col].value_counts(normalize=True).mul(100).reset_index(name='share')
    sns.barplot(
        data=df_share,
        x=col, y='share',
        order=order,
        color='salmon',
        ax=ax
    )
    ax.set_title(f'Вклад категории {col} в положительный отклик', fontsize=14)
    ax.set_ylabel('Count', fontsize = 12)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=10, rotation = 90)
    for c in ax.containers:
       ax.bar_label(c, fmt='%.1f%%', fontsize=10)
    filename = f'positive_barplot_{col}.png'
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    st.image(filename, width=500)


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Cargar los datos directamente desde la planilla adjunta (hoja "1")
# La planilla ya contiene retornos diarios (%) por clase de activo,
data1 = pd.read_excel(r"COLOCA AQUÍ LA RUTA DE TU PC QUE DEJASTE EL ARCHIVO Data_DFR", sheet_name="1")

# Convertir la columna "Date" en índice de fecha y hora
data1["Date"] = pd.to_datetime(data1["Date"])
data1.set_index("Date", inplace=True)

# dropna remove rows (or columns) from a DataFrame that contain missing values
data1.dropna(inplace=True)



## Naming the X and Y of the regression

# Modelo Clases de Activos
X_var = data1.iloc[:, 0:10]
X1 = sm.add_constant(X_var)
Y_AFP_A = data1["AFP_A"]


## Fitting the overall model
# Modelo Clases de Activos
overall_model = sm.OLS(Y_AFP_A, X1)
overall_results = overall_model.fit()
print(overall_results.summary())


## Factor decomposition across the entire period
## Solo se conservan factores:
##   (1) estadísticamente significativos (p < 0.05),
##   (2) con coeficiente positivo, 
##   (3) se normalizan para que sumen 1 (100%).


# Modelo Clases de Activos
def plot_key_factors(overall_results, alpha=0.05):
    output_df = pd.DataFrame(dict(overall_results.params), index=['Coefficients'])  # Organiza los coeficientes en un dataframe
    output_df = output_df.T
    output_df['P-values'] = overall_results.pvalues                                  # Empaqueta los p-values en el dataframe

    # Excluir la constante: no es un factor de retorno, es el intercepto del modelo
    output_df = output_df.drop(index='const', errors='ignore')

    # Filtro 1: estadísticamente significativos (p < 0.05)
    output_df = output_df[output_df['P-values'] < alpha]

    # Filtro 2: coeficiente positivo (se excluyen exposiciones negativas)
    output_df = output_df[output_df['Coefficients'] > 0]

    if output_df.empty:
        raise ValueError(
            "Ningún factor cumple simultáneamente las condiciones de significancia (p<0.05) "
            "y coeficiente positivo. Revisa el modelo o el umbral de significancia."
        )

    # Filtro/ajuste 3: normalizar los coeficientes remanentes para que sumen 1 (100%)
    output_df['Coefficients'] = output_df['Coefficients'] / output_df['Coefficients'].sum()
    key_factors = list(output_df.index)
    output_df.Coefficients.sort_values(ascending=True).plot(
        kind='barh', figsize=(10, 6), color='grey',
        title="Asset Class Factor Return Decomposition\n(significativos, positivos, normalizados a 100%)"
    ).set_xlabel("Peso normalizado")

    print("\nFactores significativos y positivos (pesos normalizados a 100%):")
    print(output_df[['Coefficients', 'P-values']])
    print("Suma de pesos:", round(output_df['Coefficients'].sum(), 6))

    return key_factors

key_factors = plot_key_factors(overall_results)


##
## Factor decomposition over time
## Cada ventana móvil también se restringe a pesos positivos que sumen 1 (100%).
##
# Modelo Clases de Activos
def factors_over_time(X_df, Y_df, window_size=500):
    coefficients_list = []
    dates_list = []
    ### Using a loop to slide a window across time to examine factor loading changes ###
    for start in range(window_size, len(Y_df) + 1):
        X_window = X_df.iloc[start - window_size:start]
        Y_window = Y_df.iloc[start - window_size:start]
        X1_window = sm.add_constant(X_window)
        window_model = sm.OLS(Y_window, X1_window)
        params = window_model.fit().params.drop(labels='const', errors='ignore')

        # Forzar solo pesos positivos (negativos -> 0) y normalizar la ventana a 1 (100%)
        weights = params.clip(lower=0)
        total = weights.sum()
        if total > 0:
            weights = weights / total

        coefficients_list.append(weights)
        dates_list.append(X_window.index[-1])
    ### Present data in a dataframe ###
    factors_loading_df = pd.DataFrame(coefficients_list)
    factors_loading_df['Dates'] = dates_list
    factors_loading_df.index = pd.to_datetime(factors_loading_df.Dates)
    factors_loading_df.set_index('Dates', inplace=True)
    return factors_loading_df

factors_loading_df = factors_over_time(data1[key_factors], data1["AFP_A"])
factors_loading_df

def plot_factor_decomposition(factors_loading_df, key_factors):
    to_plot_df = factors_loading_df.copy()
    to_plot_df['Date'] = to_plot_df.index.year.astype(str).str.cat(to_plot_df.index.month.astype(str), sep='-')
    cols_to_plot = list(key_factors)  # copia local para no mutar la lista original
    ax = to_plot_df[cols_to_plot + ["Date"]].plot(
        x='Date', y=cols_to_plot, kind='bar', stacked=True, figsize=(10, 6),
        title="Asset Class Factor Return Decomposition (suma 100%)"
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=12))
    plt.ylabel('Peso normalizado (%)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

plot_factor_decomposition(factors_loading_df['2003-01-02':], key_factors)
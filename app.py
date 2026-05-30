import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

# Configuración visual profesional
st.set_page_config(page_title="DISTRI-FIT", layout="wide", page_icon="📊")

st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>📊 SOFTWARE INDUSTRIAL 'DISTRI-FIT'</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic; color: #4b5563;'>Plataforma Grupal de Ajuste Estadístico y Simulación Monte Carlo</p>", unsafe_allow_html=True)

# Módulo de carga
st.sidebar.header("📥 1. Carga de Muestras")
archivo = st.sidebar.file_uploader("Sube un archivo Excel (.xlsx) o CSV (.csv)", type=["csv", "xlsx"])

if archivo is not None:
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        columna = st.sidebar.selectbox("Selecciona la columna con los datos reales:", df.columns)
        datos = df[columna].dropna().values
        
        if len(datos) >= 5:
            # Indicadores Descriptivos
            st.subheader("📋 Métricas de la Muestra Empírica")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Muestra (n)", len(datos))
            c2.metric("Media Muestral (μ)", f"{np.mean(datos):.3f}")
            c3.metric("Desviación (σ)", f"{np.std(datos):.3f}")
            c4.metric("Coef. Asimetría", f"{stats.skew(datos):.3f}")
            
            # Ajustes Estadísticos Simultáneos
            st.subheader("🏆 Tabla de Posiciones de Ajuste (Prueba Kolmogorov-Smirnov)")
            distribuciones = {"Normal": stats.norm, "Log-Normal": stats.lognorm, "Weibull": stats.weibull_min, "Exponencial": stats.expon}
            resultados, ajustes = [], {}
            
            for nombre, dist in distribuciones.items():
                try:
                    params = dist.fit(datos)
                    ajustes[nombre] = params
                    d_stat, p_valor = stats.kstest(datos, dist.name, args=params)
                    resultados.append({
                        "Distribución": nombre,
                        "Error K-S (Menor es mejor)": round(float(d_stat), 4),
                        "Confianza (p-valor)": round(float(p_valor), 4),
                        "Dictamen Técnico": "✅ Ajuste Óptimo" if p_valor > 0.05 else "⚠️ Ajuste Débil"
                    })
                except:
                    continue
            
            df_res = pd.DataFrame(resultados).sort_values(by="Confianza (p-valor)", ascending=False)
            st.dataframe(df_res, use_container_width=True)
            
            # Gráfico de la distribución ganadora
            dist_ganadora = df_res.iloc[0]["Distribución"]
            st.success(f"🎯 La distribución que mejor modela tus datos es: **{dist_ganadora}**")
            
            fig = px.histogram(df, x=columna, nbins=30, density="density", opacity=0.5, title="Histograma Real vs. Curva Teórica Ajustada", color_discrete_sequence=['#64748b'])
            x_axis = np.linspace(min(datos), max(datos), 100)
            y_axis = distribuciones[dist_ganadora].pdf(x_axis, *ajustes[dist_ganadora])
            fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', name=f'Modelo {dist_ganadora}', line=dict(color='#1e3a8a', width=3)))
            st.plotly_chart(fig, use_container_width=True)
            
            # PLUS MÓDULO: Monte Carlo Integrado
            st.markdown("---")
            st.subheader("🚀 Módulo Avanzado: Generador de Datos Sintéticos (Monte Carlo)")
            n_sim = st.number_input("¿Cuántos lotes nuevos deseas simular con este comportamiento?", min_value=10, max_value=5000, value=500)
            
            if st.button("▶ REPLICAR PROCESO EN LA NUBE"):
                simulados = distribuciones[dist_ganadora].rvs(*ajustes[dist_ganadora], size=n_sim)
                df_sim = pd.DataFrame({"Datos_Simulados_MonteCarlo": simulados})
                st.write("📊 **Muestra de los nuevos datos generados:**")
                st.dataframe(df_sim.head(8), use_container_width=True)
                st.download_button("💾 Descargar Datos Similados (.csv)", df_sim.to_csv(index=False).encode('utf-8'), "simulacion_distrifit.csv", "text/csv")
        else:
            st.warning("El archivo requiere al menos 5 registros numéricos.")
    except Exception as e:
        st.error(f"Error analítico: {e}")
else:
    st.info("👋 Sistema DISTRI-FIT en línea. Sube un archivo con datos reales en la barra lateral para iniciar el ajuste automático.")

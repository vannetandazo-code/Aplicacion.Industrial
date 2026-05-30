import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

# Configuración visual de alta gama corporativa
st.set_page_config(page_title="DISTRI-FIT PRO", layout="wide", page_icon="📊")

# Estilos de encabezado industrial
st.markdown("""
    <div style='background-color:#0f172a; padding:25px; border-radius:12px; margin-bottom:25px; border-left: 8px solid #3b82f6;'>
        <h1 style='color:white; margin:0; text-align:center; font-family:sans-serif;'>📊 SOFTWARE INDUSTRIAL "DISTRI-FIT PRO"</h1>
        <p style='color:#94a3b8; margin:8px 0 0 0; text-align:center; font-style:italic; font-size:12pt;'>
            Plataforma Grupal Avanzada de Ajuste Multidistribución, Auditoría Forense SPC y Simulación Estocástica
        </p>
    </div>
""", unsafe_allow_html=True)

# Módulo de carga en la barra lateral
st.sidebar.markdown("### 📥 1. Entrada de Datos")
archivo = st.sidebar.file_uploader("Sube tu muestra industrial (.xlsx) o (.csv)", type=["csv", "xlsx"])

if archivo is not None:
    try:
        # Lectura automatizada
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        columna = st.sidebar.selectbox("Selecciona la columna con los datos reales:", df.columns)
        datos = df[columna].dropna().values
        
        if len(datos) >= 5:
            # ---------------------------------------------------------
            # PESTAÑAS PRINCIPALES
            # ---------------------------------------------------------
            tab1, tab2, tab3 = st.tabs(["📊 Ajuste Estadístico y SPC", "🎯 Análisis de Riesgo y Recomendaciones", "🚀 Simulación Monte Carlo"])
            
            with tab1:
                st.subheader("📋 Métricas Descriptivas de la Muestra")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Muestra Total (n)", len(datos))
                c2.metric("Media Muestral (μ)", f"{np.mean(datos):.4f}")
                c3.metric("Desviación (σ)", f"{np.std(datos):.4f}")
                
                asimetria_val = stats.skew(datos)
                c4.metric("Coef. Asimetría", f"{asimetria_val:.4f}")
                
                st.markdown("---")
                st.subheader("🏆 Tabla de Posiciones de Ajuste (Prueba Kolmogorov-Smirnov)")
                
                distribuciones = {
                    "Normal (Gaussiana)": stats.norm, 
                    "Log-Normal": stats.lognorm, 
                    "Weibull (Fallas/Tiempos)": stats.weibull_min, 
                    "Exponencial": stats.expon,
                    "Gamma": stats.gamma
                }
                
                resultados, ajustes = [], {}
                for nombre, dist in distribuciones.items():
                    try:
                        params = dist.fit(datos)
                        ajustes[nombre] = params
                        d_stat, p_valor = stats.kstest(datos, dist.name, args=params)
                        resultados.append({
                            "Distribución Evaluada": nombre,
                            "Error K-S (Menor es mejor)": round(float(d_stat), 5),
                            "Confianza (p-valor)": round(float(p_valor), 5),
                            "Dictamen Técnico": "✅ Ajuste Óptimo (Aceptado)" if p_valor > 0.05 else "⚠️ Ajuste Débil"
                        })
                    except:
                        continue
                
                df_res = pd.DataFrame(resultados).sort_values(by="Confianza (p-valor)", ascending=False)
                st.dataframe(df_res, use_container_width=True)
                
                dist_ganadora = df_res.iloc[0]["Distribución Evaluada"]
                st.success(f"🎯 **Conclusión Estadística:** La distribución ganadora que mejor modela este proceso es la **{dist_ganadora}**.")
                
                # Desglose de Parámetros Técnicos
                st.subheader("🧬 Parámetros de Diseño Descubiertos (MLE)")
                params_ganadores = ajustes[dist_ganadora]
                
                cp1, cp2 = st.columns(2)
                with cp1:
                    st.info(f"**Valores del Ajuste:**\n\n{list(np.round(params_ganadores, 4))}")
                with cp2:
                    st.write("📝 *Nota para simulación:* Copia estos valores exactos en tus modelos de simulación de eventos discretos para replicar el comportamiento real de la planta de forma matemática.")
                
                # Gráfica interactiva 1: Histograma vs Curva Teórica
                st.markdown("---")
                st.subheader("📈 Verificación Gráfica del Acoplamiento")
                fig = px.histogram(df, x=columna, nbins=30, histnorm='probability density', 
                                   opacity=0.5, title=f"Histograma Real vs. Modelo Continuo {dist_ganadora}", 
                                   color_discrete_sequence=['#475569'])
                
                x_axis = np.linspace(min(datos), max(datos), 200)
                y_axis = distribuciones[dist_ganadora].pdf(x_axis, *params_ganadores)
                fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', name=dist_ganadora, line=dict(color='#3b82f6', width=3.5)))
                st.plotly_chart(fig, use_container_width=True)

                # INTERPRETACIÓN AUTOMÁTICA DEL HISTOGRAMA
                st.markdown("🔍 **Interpretación del Histograma:**")
                if abs(asimetria_val) < 0.5:
                    st.info(f"El gráfico muestra un comportamiento **simétrico** (forma de campana equilibrada). Esto significa que la mayoría de los eventos ocurren muy cerca del promedio ({np.mean(datos):.2f}) y las variaciones extremas hacia arriba o hacia abajo son igualmente raras. Al ser modelado óptimamente por una distribución **{dist_ganadora}**, indica que no existen factores externos sesgando el proceso de manera severa.")
                elif asimetria_val >= 0.5:
                    st.info(f"El gráfico presenta un **sesgo positivo (hacia la derecha)**. Esto significa que el proceso tiende a acumular la mayoría de sus datos en valores bajos, pero tiene una 'cola larga' de eventos con valores inusualmente altos. En entornos de manufactura, esto delata la presencia recurrente de retrasos, cuellos de botella u operaciones ineficientes que alargan los tiempos de forma atípica.")
                else:
                    st.info(f"El gráfico presenta un **sesgo negativo (hacia la izquierda)**. Esto indica que la muestra tiende a acumularse en los valores más altos, cayendo abruptamente en los rangos bajos. Delata procesos donde existe un límite superior físico restrictivo o un comportamiento de desgaste acelerado antes de estabilizarse.")

                # GRÁFICO 2: Control Estadístico de Procesos (SPC)
                st.markdown("---")
                st.subheader("📉 Auditoría Forense: Gráfico de Control y Tendencia del Proceso (SPC)")
                st.write("Visualiza la evolución temporal de la variable e identifica rachas fuera de estabilidad:")
                
                indices = np.arange(1, len(datos) + 1)
                fig_spc = go.Figure()
                
                # Línea de corridas reales
                fig_spc.add_trace(go.Scatter(x=indices, y=datos, mode='lines+markers', name='Valor Registrado', line=dict(color='#64748b', width=1.5)))
                # Línea de Promedio
                fig_spc.add_trace(go.Scatter(x=indices, y=[np.mean(datos)]*len(datos), mode='lines', name='Promedio Muestral', line=dict(color='#16a34a', dash='dash', width=2)))
                # Línea de Máximo histórico
                fig_spc.add_trace(go.Scatter(x=indices, y=[np.max(datos)]*len(datos), mode='lines', name='Máximo Registrado', line=dict(color='#dc2626', dash='dot', width=1.5)))
                # Línea de Mínimo histórico
                fig_spc.add_trace(go.Scatter(x=indices, y=[np.min(datos)]*len(datos), mode='lines', name='Mínimo Registrado', line=dict(color='#2563eb', dash='dot', width=1.5)))
                
                fig_spc.update_layout(
                    title="Evolución Cronológica del Proceso y Líneas de Límites Físicos", 
                    xaxis_title="Orden de Corrida / Lote", 
                    yaxis_title="Magnitud Medida"
                )
                st.plotly_chart(fig_spc, use_container_width=True)

                # INTERPRETACIÓN AUTOMÁTICA DEL GRÁFICO SPC
                st.markdown("🔍 **Interpretación Forense de Tendencias (SPC):**")
                # Detección simple de picos repetidos cerca del máximo
                valores_altos = np.sum(datos > (np.mean(datos) + np.std(datos)))
                porcentaje_altos = (valores_altos / len(datos)) * 100
                
                st.info(f"Este gráfico evalúa la **estabilidad temporal**. La línea central verde marca el comportamiento histórico ideal ({np.mean(datos):.2f}). Se observa que un **{porcentaje_altos:.1f}%** de los datos medidos se encuentran en la zona de alta dispersión (por encima de 1 desviación estándar). Si notas picos consecutivos tocando repetidamente la línea roja superior (Máximo: {np.max(datos):.2f}), el gráfico evidencia fallas mecánicas intermitentes o fatiga de materiales crónicos en esos lotes específicos, rompiendo la predictibilidad del sistema.")
            
            with tab2:
                # MÓDULO: Gestión y Análisis de Riesgo Operativo
                st.subheader("🎯 Auditoría de Riesgo Contractual y Capacidad de Proceso")
                st.write("Ingresa los límites específicos acordados con tu cliente para evaluar el nivel de cumplimiento del proceso:")
                
                cr1, cr2 = st.columns(2)
                with cr1:
                    limite_inferior = st.number_input("Especificación Inferior (LIE):", value=float(min(datos)))
                with cr2:
                    limite_superior = st.number_input("Especificación Superior (LSE):", value=float(max(datos)))
                
                if limite_superior > limite_inferior:
                    dist_objeto = distribuciones[dist_ganadora]
                    
                    # Cálculo de áreas bajo la curva usando la función de distribución acumulada (CDF)
                    prob_debajo = dist_objeto.cdf(limite_inferior, *params_ganadores)
                    prob_encima = 1.0 - dist_objeto.cdf(limite_superior, *params_ganadores)
                    prob_dentro = 1.0 - prob_debajo - prob_encima
                    
                    st.markdown("### 📊 Diagnóstico de Operaciones:")
                    colr1, colr2, colr3 = st.columns(3)
                    colr1.metric("Probabilidad de Aceptación (Dentro de Contrato)", f"{prob_dentro*100:.2f} %")
                    colr2.metric("Riesgo de Rechazo por Defecto (Por debajo)", f"{prob_debajo*100:.2f} %")
                    colr3.metric("Riesgo de Entrega Tardía (Por encima)", f"{prob_encima*100:.2f} %")
                    
                    st.markdown("---")
                    st.subheader("📋 Consultor de Operaciones Automatizado: Recomendaciones Técnicas")
                    
                    # Generador de Recomendaciones Inteligentes Basadas en Riesgo
                    if prob_dentro >= 0.90:
                        st.markdown("""
                            <div style='background-color:#f0fdf4; padding:20px; border-radius:8px; border-left:6px solid #16a34a;'>
                                <h4 style='color:#14532d; margin:0;'>🟢 ESTADO: PROCESO BAJO CONTROL TÉCNICO Y COMERCIAL</h4>
                                <p style='color:#166534; margin-top:10px; font-size:11pt;'>
                                    El proceso actual demuestra una capacidad sobresaliente y un nivel mínimo de pérdidas estructurales. Se recomiendan las siguientes acciones:
                                </p>
                                <ul style='color:#166534; font-size:10.5pt;'>
                                    <li><strong>Estandarización Continua:</strong> Formalizar las variables operativas de la corrida actual en guías de mejores prácticas operacionales (SOP).</li>
                                    <li><strong>Mantenimiento Preventivo Planificado:</strong> Mantener los ciclos vigentes de calibración y lubricación de equipos mecánicos para conservar este nivel de variabilidad controlada.</li>
                                    <li><strong>Alineación Comercial:</strong> Utilizar este simulador interactivo para negociar de manera segura tolerancias contractuales aún más estrictas con nuevos clientes estratégicos sin poner en peligro los márgenes financieros.</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                    elif prob_dentro >= 0.75:
                        st.markdown("""
                            <div style='background-color:#fffbeb; padding:20px; border-radius:8px; border-left:6px solid #d97706;'>
                                <h4 style='color:#78350f; margin:0;'>🟡 ESTADO: PRECAUCIÓN - PROCESO CON VARIABILIDAD AMENAZANTE</h4>
                                <p style='color:#92400e; margin-top:10px; font-size:11pt;'>
                                    La variabilidad física del sistema está comenzando a invadir las zonas de tolerancia comercial del cliente. Se requiere atención proactiva inmediata:
                                </p>
                                <ul style='color:#92400e; font-size:10.5pt;'>
                                    <li><strong>Auditoría de Cuellos de Botella Mecánicos:</strong> Inspeccionar los sistemas mecánicos auxiliares, tolvas o dosificadores para descartar atascos periódicos o microparadas mecánicas recurrentes.</li>
                                    <li><strong>Control Térmico e Instrumentación:</strong> Verificar la calibración de los sensores de temperatura y presión en las calderas o reactores para evitar que la variación cinética mueva el promedio real del proceso.</li>
                                    <li><strong>Monitoreo en Tiempo Real:</strong> Utilizar de forma diaria el Gráfico de Control SPC (Módulo 1 de esta aplicación) para identificar tendencias acumulativas antes de que los lotes comiencen a salir defectuosos de forma permanente.</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style='background-color:#fef2f2; padding:20px; border-radius:8px; border-left:6px solid #dc2626;'>
                                <h4 style='color:#7f1d1d; margin:0;'>🔴 ESTADO CRÍTICO: PROCESO REBASADO / FUERA DE CONTROL</h4>
                                <p style='color:#991b1b; margin-top:10px; font-size:11pt;'>
                                    El proceso industrial opera bajo condiciones inestables de alta merma o retrasos crónicos severos. Se exigen medidas drásticas estructurales inmediatas:
                                </p>
                                <ul style='color:#991b1b; font-size:10.5pt;'>
                                    <li><strong>Justificación de Reemplazo de Activos (CAPEX):</strong> La obsolescencia de los componentes mecánicos está destruyendo la rentabilidad. Se recomienda utilizar los reportes estadísticos de este simulador para justificar ante la dirección general la compra urgente de maquinaria automatizada de flujo continuo.</li>
                                    <li><strong>Reingeniería Operativa Inmediata:</strong> Detener temporalmente las operaciones críticas para estabilizar las condiciones cinéticas básicas del reactor y reducir drásticamente el impacto de las sub-operaciones ineficientes.</li>
                                    <li><strong>Contingencia Legal Contractual:</strong> Revisar con el equipo comercial las penalizaciones vigentes por retrasos en entregas tardías, ya que la probabilidad de fallo actual representa un riesgo comercial severo de demandas o pérdidas de contratos vigentes.</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
            
            with tab3:
                # MÓDULO 3: Generador de Datos Sintéticos Monte Carlo
                st.subheader("🚀 Generador Estocástico de Datos en la Nube")
                st.write("Genera escenarios sintéticos utilizando los parámetros exactos calculados por la aplicación:")
                
                n_sim = st.number_input("Cantidad de nuevos lotes a simular por Monte Carlo:", min_value=10, max_value=10000, value=1000, step=50)
                
                if st.button("▶ EJECUTAR SIMULACIÓN DE ESCENARIOS"):
                    dist_objeto = distribuciones[dist_ganadora]
                    simulados = dist_objeto.rvs(*params_ganadores, size=n_sim)
                    
                    # Truncamiento inferior para evitar números negativos físicamente imposibles
                    simulados = np.clip(simulados, a_min=0, a_max=None)
                    
                    df_sim = pd.DataFrame({"Datos_Simulados_MonteCarlo": simulados})
                    
                    cs1, cs2 = st.columns(2)
                    with cs1:
                        st.write("📊 **Primeras filas del nuevo juego de datos estocásticos:**")
                        st.dataframe(df_sim.head(10), use_container_width=True)
                    with cs2:
                        st.write("💾 **Panel de Exportación Directa:**")
                        st.info(f"Modelo Matemático Aplicado: {dist_ganadora}\n\nDatos Proyectados con Éxito sin valores negativos físicos.")
                        
                        # Botón para descargar los datos en formato CSV
                        csv_data = df_sim.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Descargar Simulación (.csv)", 
                                           data=csv_data, 
                                           file_name="simulacion_montecarlo_pro.csv", 
                                           mime="text/csv")
        else:
            st.warning("⚠ El archivo seleccionado no contiene suficientes datos válidos para ejecutar un análisis estadístico.")
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo analítico: {e}")
else:
    st.info("👋 Bienvenida a DISTRI-FIT PRO. Por favor, sube un archivo Excel o CSV en la barra lateral para desplegar los módulos avanzados de análisis.")

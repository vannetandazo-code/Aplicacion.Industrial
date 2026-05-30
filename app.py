import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

# Configuración visual de alta gama corporativa
st.set_page_config(page_title="DISTRI-FIT PRO v4", layout="wide", page_icon="📊")

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
            # PESTAÑAS PRINCIPALES (Actualizado a 4 pestañas estratégicas)
            # ---------------------------------------------------------
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Ajuste Estadístico y SPC", 
                "🎯 Análisis de Riesgo Contractual", 
                "🚀 Simulación Monte Carlo", 
                "📋 Plan de Acción y Siguientes Pasos"
            ])
            
            # --- PESTAÑA 1: AJUSTE Y SPC ---
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

                # Interpretación automática del Histograma
                st.markdown("🔍 **Interpretación del Histograma:**")
                if abs(asimetria_val) < 0.5:
                    st.info(f"El gráfico muestra un comportamiento **simétrico** (forma de campana equilibrada). Esto significa que la mayoría de los eventos ocurren muy cerca del promedio ({np.mean(datos):.2f}) y las variaciones extremas hacia arriba o hacia abajo son igualmente raras. Al ser modelado óptimamente por una distribución **{dist_ganadora}**, indica que no existen factores externos sesgando el proceso de manera severa.")
                elif asimetria_val >= 0.5:
                    st.info(f"El gráfico presenta un **sesgo positivo (hacia la derecha)**. Esto significa que el proceso tiende a acumular la mayoría de sus datos en valores bajos, pero tiene una 'cola larga' de eventos con valores inusualmente altos. En entornos de manufactura, esto delata la presencia recurrente de retrasos, cuellos de botella u operaciones ineficientes que alargan los tiempos de forma atípica.")
                else:
                    st.info(f"El gráfico presenta un **sesgo negativo (hacia la izquierda)**. Esto indica que la muestra tiende a acumularse en los valores más altos, cayendo abruptamente en los rangos bajos. Delata procesos donde existe un límite superior físico restrictivo o un comportamiento de desgaste acelerado antes de estabilizarse.")

                # Gráfico SPC
                st.markdown("---")
                st.subheader("📉 Auditoría Forense: Gráfico de Control y Tendencia del Proceso (SPC)")
                indices = np.arange(1, len(datos) + 1)
                fig_spc = go.Figure()
                fig_spc.add_trace(go.Scatter(x=indices, y=datos, mode='lines+markers', name='Valor Registrado', line=dict(color='#64748b', width=1.5)))
                fig_spc.add_trace(go.Scatter(x=indices, y=[np.mean(datos)]*len(datos), mode='lines', name='Promedio Muestral', line=dict(color='#16a34a', dash='dash', width=2)))
                fig_spc.add_trace(go.Scatter(x=indices, y=[np.max(datos)]*len(datos), mode='lines', name='Máximo Registrado', line=dict(color='#dc2626', dash='dot', width=1.5)))
                fig_spc.add_trace(go.Scatter(x=indices, y=[np.min(datos)]*len(datos), mode='lines', name='Mínimo Registrado', line=dict(color='#2563eb', dash='dot', width=1.5)))
                
                fig_spc.update_layout(title="Evolución Cronológica del Proceso y Líneas de Límites Físicos", xaxis_title="Orden de Corrida / Lote", yaxis_title="Magnitud Medida")
                st.plotly_chart(fig_spc, use_container_width=True)

                # Interpretación del SPC
                st.markdown("🔍 **Interpretación Forense de Tendencias (SPC):**")
                valores_altos = np.sum(datos > (np.mean(datos) + np.std(datos)))
                porcentaje_altos = (valores_altos / len(datos)) * 100
                st.info(f"Este gráfico evalúa la **estabilidad temporal**. La línea central verde marca el comportamiento histórico ideal ({np.mean(datos):.2f}). Se observa que un **{porcentaje_altos:.1f}%** de los datos medidos se encuentran en la zona de alta dispersión (por encima de 1 desviación estándar). Si notas picos consecutivos tocando repetidamente la línea roja superior (Máximo: {np.max(datos):.2f}), el gráfico evidencia fallas mecánicas intermitentes o fatiga de materiales crónicos en esos lotes específicos, rompiendo la predictibilidad del sistema.")
            
            # --- PESTAÑA 2: RIESGO ---
            with tab2:
                st.subheader("🎯 Auditoría de Riesgo Contractual y Capacidad de Proceso")
                st.write("Ingresa los límites específicos acordados con tu cliente para evaluar el nivel de cumplimiento del proceso:")
                
                cr1, cr2 = st.columns(2)
                with cr1:
                    limite_inferior = st.number_input("Especificación Inferior (LIE):", value=float(min(datos)))
                with cr2:
                    limite_superior = st.number_input("Especificación Superior (LSE):", value=float(max(datos)))
                
                if limite_superior > limite_inferior:
                    dist_objeto = distribuciones[dist_ganadora]
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
                    
                    if prob_dentro >= 0.90:
                        estado_actual = "VERDE"
                        st.markdown("""
                            <div style='background-color:#f0fdf4; padding:20px; border-radius:8px; border-left:6px solid #16a34a;'>
                                <h4 style='color:#14532d; margin:0;'>🟢 ESTADO: PROCESO BAJO CONTROL TÉCNICO Y COMERCIAL</h4>
                            </div>
                        """, unsafe_allow_html=True)
                    elif prob_dentro >= 0.75:
                        estado_actual = "AMARILLO"
                        st.markdown("""
                            <div style='background-color:#fffbeb; padding:20px; border-radius:8px; border-left:6px solid #d97706;'>
                                <h4 style='color:#78350f; margin:0;'>🟡 ESTADO: PRECAUCIÓN - PROCESO CON VARIABILIDAD AMENAZANTE</h4>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        estado_actual = "ROJO"
                        st.markdown("""
                            <div style='background-color:#fef2f2; padding:20px; border-radius:8px; border-left:6px solid #dc2626;'>
                                <h4 style='color:#7f1d1d; margin:0;'>🔴 ESTADO CRÍTICO: PROCESO REBASADO / FUERA DE CONTROL</h4>
                            </div>
                        """, unsafe_allow_html=True)
            
            # --- PESTAÑA 3: MONTE CARLO ---
            with tab3:
                st.subheader("🚀 Generador Estocástico de Datos en la Nube")
                n_sim = st.number_input("Cantidad de nuevos lotes a simular por Monte Carlo:", min_value=10, max_value=10000, value=1000, step=50)
                
                if st.button("▶ EJECUTAR SIMULACIÓN DE ESCENARIOS"):
                    dist_objeto = distribuciones[dist_ganadora]
                    simulados = dist_objeto.rvs(*params_ganadores, size=n_sim)
                    simulados = np.clip(simulados, a_min=0, a_max=None)
                    df_sim = pd.DataFrame({"Datos_Simulados_MonteCarlo": simulados})
                    
                    cs1, cs2 = st.columns(2)
                    with cs1:
                        st.write("📊 **Primeras filas del nuevo juego de datos estocásticos:**")
                        st.dataframe(df_sim.head(10), use_container_width=True)
                    with cs2:
                        st.write("💾 **Panel de Exportación Directa:**")
                        st.info(f"Modelo Matemático Aplicado: {dist_ganadora}\n\nDatos Proyectados con Éxito sin valores negativos físicos.")
                        csv_data = df_sim.to_csv(index=False).encode('utf-8')
                        st.download_button(label="📥 Descargar Simulación (.csv)", data=csv_data, file_name="simulacion_montecarlo_pro.csv", mime="text/csv")
            
            # --- NUEVA PESTAÑA 4: PLAN DE ACCIÓN Y RECOMENDACIONES DE USO ---
            with tab4:
                st.header("📋 Hoja de Ruta Consultora y Casos de Uso")
                st.write("A continuación, se detalla qué decisiones estratégicas de ingeniería puedes tomar utilizando el perfil probabilístico descubierto:")
                
                c_act1, c_act2 = st.columns(2)
                
                with c_act1:
                    st.markdown("""
                        <div style='background-color:#f8fafc; padding:20px; border-radius:10px; border:1px solid #e2e8f0;'>
                            <h3 style='color:#0f172a; margin-top:0;'>🛠️ 1. Siguientes Pasos Operativos (Roadmap)</h3>
                            <ol style='font-size:11pt; color:#334155; line-height:1.6;'>
                                <li><strong>Ingresar Parámetros en Simuladores:</strong> Exporta los parámetros calculados (MLE) en softwares de simulación de planta (FlexSim, Arena o Promodel) para modelar escenarios reales de cuellos de botella sin sesgos humanos.</li>
                                <li><strong>Rediseño de Capacidad Comercial:</strong> Utiliza el porcentaje de riesgo hallado en la Pestaña 2 para renegociar con los clientes plazos de entrega más realistas o tarifas de penalización justas.</li>
                                <li><strong>Ajuste de Alarmas en Maquinaria:</strong> Configura los límites calculados en el gráfico SPC dentro de los sensores SCADA de la planta para que emitan alertas preventivas automáticas antes de que un lote se desvíe.</li>
                            </ol>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Recomendación dinámica según el estado del semáforo
                    st.markdown("### ⚠️ Acción Prioritaria Recomendada:")
                    if prob_dentro >= 0.90:
                        st.success("✨ **Recomendación Directiva:** Tu proceso es altamente confiable. Tu siguiente paso estratégico debe ser la **reducción de costos de inspección**. Al saber que la probabilidad de fallo es mínima, puedes disminuir el muestreo destructivo manual y confiar plenamente en el control preventivo básico.")
                    elif prob_dentro >= 0.75:
                        st.warning("⚡ **Recomendación Directiva:** Tu proceso está en zona de riesgo moderado. Tu siguiente paso prioritario debe ser la **estandarización de turnos**. El gráfico SPC denota variación por lotes; audita si el cambio de operarios o proveedores de materia prima está alterando la uniformidad del sistema.")
                    else:
                        st.error("🚨 **Recomendación Directiva:** El proceso actual está colapsado. Tu siguiente paso de inversión debe ser la **automatización o reingeniería de maquinaria**. Los datos empíricos demuestran que las tolerancias requeridas están fuera del alcance físico actual de tu equipo.")

                with c_act2:
                    st.markdown("""
                        <div style='background-color:#f1f5f9; padding:20px; border-radius:10px; border:1px solid #cbd5e1;'>
                            <h3 style='color:#1e293b; margin-top:0;'>💡 2. ¿Cómo usar estos datos en otras áreas?</h3>
                            <ul style='font-size:10.5pt; color:#475569; line-height:1.6;'>
                                <li><strong>Área de Finanzas (Costeo de Mermas):</strong> Multiplica el porcentaje de riesgo de fallo por el costo de fabricación de un lote. Esto le dará al departamento financiero el presupuesto exacto de pérdidas anuales proyectadas.</li>
                                <li><strong>Área de Logística y Cadena de Suministro:</strong> El simulador Monte Carlo te permite planificar cuánta materia prima necesitarás almacenar en bodega para absorber los retrasos o picos que la distribución continua ya predijo.</li>
                                <li><strong>Departamento de Mantenimiento:</strong> Si la distribución ganadora fue una <em>Weibull</em> o <em>Gamma</em>, los ingenieros mecánicos pueden calcular el tiempo óptimo entre fallas (MTBF) para programar reparaciones antes de que la máquina colapse por completo.</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Bloque pedagógico sobre cómo escribir la fórmula en Arena o FlexSim
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 💻 Expresión Matemática para Software de Eventos Discretos:")
                    st.write("Si necesitas simular este proceso en otros entornos de ingeniería, la sintaxis correcta que debes introducir es:")
                    
                    clean_params = [round(p, 3) for p in params_ganadores]
                    if "Normal" in dist_ganadora:
                        st.code(f"NORM({clean_params[0]}, {clean_params[1]})", language="python")
                    elif "Log-Normal" in dist_ganadora:
                        st.code(f"LOGN({clean_params[0]}, {clean_params[1]})", language="python")
                    elif "Weibull" in dist_ganadora:
                        st.code(f"WEIB({clean_params[0]}, {clean_params[1]})", language="python")
                    elif "Exponencial" in dist_ganadora:
                        st.code(f"EXPO({clean_params[0]})", language="python")
                    else:
                        st.code(f"GAMM({clean_params[0]}, {clean_params[1]})", language="python")
                        
        else:
            st.warning("⚠ El archivo seleccionado no contiene suficientes datos válidos para ejecutar un análisis estadístico.")
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo analítico: {e}")
else:
    st.info("👋 Bienvenida a DISTRI-FIT PRO. Por favor, sube un archivo Excel o CSV en la barra lateral para desplegar los módulos avanzados de análisis.")

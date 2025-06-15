from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import random
from datetime import datetime, timedelta
import numpy as np

# Cargar modelo y herramientas (si las necesitas para otros endpoints)
modelo = joblib.load("modelo_rf.pkl")
encoders = joblib.load("encoders.pkl")
columnas = joblib.load("columnas_modelo.pkl")

# Cargar DataFrame completo con predicciones y características
df_clientes_completo = pd.read_csv("todos_los_datos.csv", encoding="latin-1")

# --- FUNCIÓN PARA GENERAR RETO PERSONALIZADO ---
def generar_reto_para_cliente(cliente_id, df_clientes_completo_param, temporada_actual="General"):
    cliente_data = df_clientes_completo_param[df_clientes_completo_param['ID Cliente'] == cliente_id]

    if cliente_data.empty:
        return None

    cliente_data = cliente_data.iloc[0]

    fecha_inicio_reto = datetime.now()
    fecha_fin_reto = fecha_inicio_reto + timedelta(days=random.randint(15, 30))

    reto = {
        "ID_Cliente": int(cliente_id),
        "Nombre_Reto": "",
        "Descripcion": "",
        "Puntos_Recompensa": 0,
        "Vigencia_Inicio": fecha_inicio_reto.strftime('%Y-%m-%d'),
        "Vigencia_Fin": fecha_fin_reto.strftime('%Y-%m-%d'),
        "Tipo_Reto": "",
        "Categoria_Enfocada": "General",
        "Producto_Sugerido": "Variedad Coca-Cola",
        "Mensaje_Emocional": ""
    }

    is_alto_valor = cliente_data['EsClienteAltoValor_pred'] if 'EsClienteAltoValor_pred' in cliente_data else 0
    prob_alto_valor = cliente_data['ProbAltoValor'] if 'ProbAltoValor' in cliente_data else 0.5

    categorias_ventas = {}
    for col in df_clientes_completo_param.columns:
        if col.startswith('Ventas_') and col in cliente_data:
            categoria_nombre = col.replace('Ventas_', '').replace('_', ' ')
            categorias_ventas[categoria_nombre] = cliente_data[col]

    categoria_mas_comprada = "Bebidas Refrescantes"
    categoria_menos_comprada = "Otras Bebidas"

    if categorias_ventas:
        categorias_con_ventas = {k: v for k, v in categorias_ventas.items() if v > 0}
        categorias_sin_ventas = [k for k, v in categorias_ventas.items() if v == 0]

        if categorias_con_ventas:
            categoria_mas_comprada = max(categorias_con_ventas, key=categorias_con_ventas.get)
            if categorias_sin_ventas:
                categoria_menos_comprada = random.choice(categorias_sin_ventas)
            else:
                categoria_menos_comprada = min(categorias_con_ventas, key=categorias_con_ventas.get)
        elif categorias_sin_ventas:
            categoria_menos_comprada = random.choice(categorias_sin_ventas)

    preferencia_sabor = cliente_data.get('Preferencia_Sabor', 'Dulce')
    momento_consumo = cliente_data.get('Momento_Consumo_Preferido', 'Comidas')
    sensibilidad_precio = cliente_data.get('Sensibilidad_Precio', 'Media')

    monto_gasto_promedio_x_compra = cliente_data.get('valor_monetario_total', 100) / cliente_data.get('frecuencia_compras', 1) if cliente_data.get('frecuencia_compras', 1) > 0 else 100

    reto_mensajes = {
        "Alto_Valor": {
            "Recompensa por Lealtad": {
                "Nombre": f"¡Tu Lealtad Refresca Más, Cliente Premium!",
                "Descripcion": lambda cat_mas: f"Disfruta aún más de lo que amas. Por tu lealtad, recibe puntos extra en tu próxima compra de {cat_mas}. ¡Celebra cada momento con Coca-Cola!",
                "Puntos": lambda venta_esp: max(75, min(250, int(venta_esp * 0.007))),
                "Producto": "Tu Favorito",
                "Emocional": "¡Eres parte esencial de la familia Coca-Cola!"
            },
            "Explora Nueva Categoría": {
                "Nombre": lambda cat_menos: f"Expande tu Sabor: ¡Descubre {cat_menos}!",
                "Descripcion": lambda cat_menos, monto: f"Siempre hay algo nuevo para probar. Explora nuestra variedad de {cat_menos} y realiza una compra de ${monto:.2f} USD para ganar puntos. ¿Listo para la aventura de sabor?",
                "Puntos": lambda venta_esp: int(venta_esp * 0.025),
                "Producto": lambda cat_menos: f"Nuevos sabores en {cat_menos}",
                "Emocional": "¡Despierta tu curiosidad y sorprende a tu paladar!"
            },
            "Compra de Mayor Volumen": {
                "Nombre": f"¡Maximiza tu Sabor, Multiplica tus Puntos!",
                "Descripcion": lambda monto: f"Para tus grandes momentos, grandes recompensas. Alcanza una compra de ${monto:.2f} USD este mes y dobla tus puntos. ¡Ideal para compartir!",
                "Puntos": lambda venta_esp: int(venta_esp * 0.02),
                "Producto": "Paquetes Familiares/Fiesta",
                "Emocional": "¡Que nada te falte para celebrar tus éxitos!"
            }
        },
        "Bajo_Valor": {
            "Primera Compra en Categoría": {
                "Nombre": lambda cat_menos: f"¡Tu Primera Chispa en {cat_menos} y Duplica tus Puntos!",
                "Descripcion": lambda cat_menos, monto: f"Un nuevo mundo de frescura te espera. Prueba nuestra categoría de {cat_menos} con una compra mínima de ${monto:.2f} USD y **duplica tus puntos de bienvenida**.",
                "Puntos": lambda venta_esp: int(venta_esp * 0.04),
                "Producto": lambda cat_menos: f"Producto estrella de {cat_menos}",
                "Emocional": "¡Anímate a probar y te sorprenderás con el doble de recompensa!"
            },
            "Aumenta tu Frecuencia": {
                "Nombre": f"¡Más Veces, Más Refresco, DOBLE de Puntos!",
                "Descripcion": lambda freq: f"Queremos verte más seguido. Realiza {freq} compras este mes y **duplica los puntos** acumulados. ¡Cada momento con Coca-Cola es especial!",
                "Puntos": lambda venta_esp: int(venta_esp * 0.04),
                "Producto": "Bebida Individual",
                "Emocional": "¡Haz de Coca-Cola parte de tu día a día y gana el doble!"
            },
            "Compra Mínima Garantizada": {
                "Nombre": f"¡Tu Momento Refrescante a un Paso y DOBLE Recompensa!",
                "Descripcion": lambda monto: f"Un pequeño gasto, una gran recompensa. Realiza una compra de al menos ${monto:.2f} USD este mes y recibe el **doble de puntos** extra para tus próximas recompensas.",
                "Puntos": lambda venta_esp: int(venta_esp * 0.04),
                "Producto": "Snack y Bebida",
                "Emocional": "¡Refresca tus ideas, impulsa tu día con el doble de beneficios!"
            }
        }
    }

    # --- Selección de Reto y Personalización Fina ---
    if is_alto_valor == 1:
        grupo_reto = reto_mensajes["Alto_Valor"]
        tipo_reto_elegido = random.choice(list(grupo_reto.keys()))
        info_reto = grupo_reto[tipo_reto_elegido]
    else:
        grupo_reto = reto_mensajes["Bajo_Valor"]
        tipo_reto_elegido = random.choice(list(grupo_reto.keys()))
        info_reto = grupo_reto[tipo_reto_elegido]

    reto["Tipo_Reto"] = tipo_reto_elegido

    venta_esperada_reto = 0

    if tipo_reto_elegido == "Recompensa por Lealtad":
        venta_esperada_reto = cliente_data.get('valor_monetario_total', 100)
        reto["Nombre_Reto"] = info_reto["Nombre"]
        reto["Descripcion"] = info_reto["Descripcion"](categoria_mas_comprada)
        reto["Puntos_Recompensa"] = info_reto["Puntos"](venta_esperada_reto)
        reto["Categoria_Enfocada"] = categoria_mas_comprada
        reto["Producto_Sugerido"] = info_reto["Producto"]
        reto["Mensaje_Emocional"] = info_reto["Emocional"]

    elif tipo_reto_elegido == "Explora Nueva Categoría":
        venta_esperada_reto = 50
        reto["Nombre_Reto"] = info_reto["Nombre"](categoria_menos_comprada)
        reto["Descripcion"] = info_reto["Descripcion"](categoria_menos_comprada, venta_esperada_reto)
        reto["Puntos_Recompensa"] = info_reto["Puntos"](venta_esperada_reto)
        reto["Categoria_Enfocada"] = categoria_menos_comprada
        reto["Producto_Sugerido"] = info_reto["Producto"](categoria_menos_comprada)
        reto["Mensaje_Emocional"] = info_reto["Emocional"]

    elif tipo_reto_elegido == "Compra de Mayor Volumen":
        target_aumento_porcentaje = random.uniform(0.1, 0.2)
        monto_objetivo = monto_gasto_promedio_x_compra * (1 + target_aumento_porcentaje)
        monto_objetivo = max(100, monto_objetivo)
        venta_esperada_reto = monto_objetivo
        reto["Nombre_Reto"] = info_reto["Nombre"]
        reto["Descripcion"] = info_reto["Descripcion"](monto_objetivo)
        reto["Puntos_Recompensa"] = info_reto["Puntos"](venta_esperada_reto)
        reto["Categoria_Enfocada"] = "General"
        reto["Producto_Sugerido"] = info_reto["Producto"]
        reto["Mensaje_Emocional"] = info_reto["Emocional"]

    elif tipo_reto_elegido == "Primera Compra en Categoría":
        venta_esperada_reto = 30
        reto["Nombre_Reto"] = info_reto["Nombre"](categoria_menos_comprada)
        reto["Descripcion"] = info_reto["Descripcion"](categoria_menos_comprada, venta_esperada_reto)
        reto["Puntos_Recompensa"] = info_reto["Puntos"](venta_esperada_reto)
        reto["Categoria_Enfocada"] = categoria_menos_comprada
        reto["Producto_Sugerido"] = info_reto["Producto"](categoria_menos_comprada)
        reto["Mensaje_Emocional"] = info_reto["Emocional"]

    elif tipo_reto_elegido == "Aumenta tu Frecuencia":
        frecuencia_actual = cliente_data.get('frecuencia_compras', 1)
        target_frecuencia = frecuencia_actual + 1 if frecuencia_actual > 0 else 2
        venta_esperada_reto = monto_gasto_promedio_x_compra * target_frecuencia
        venta_esperada_reto = max(50, venta_esperada_reto)
        reto["Nombre_Reto"] = info_reto["Nombre"]
        reto["Descripcion"] = info_reto["Descripcion"](target_frecuencia)
        reto["Puntos_Recompensa"] = info_reto["Puntos"](venta_esperada_reto)
        reto["Categoria_Enfocada"] = "General"
        reto["Producto_Sugerido"] = info_reto["Producto"]
        reto["Mensaje_Emocional"] = info_reto["Emocional"]

    elif tipo_reto_elegido == "Compra Mínima Garantizada":
        monto_minimo = 75
        venta_esperada_reto = monto_minimo
        reto["Nombre_Reto"] = info_reto["Nombre"]
        reto["Descripcion"] = info_reto["Descripcion"](monto_minimo)
        reto["Puntos_Recompensa"] = info_reto["Puntos"](venta_esperada_reto)
        reto["Categoria_Enfocada"] = "General"
        reto["Producto_Sugerido"] = info_reto["Producto"]
        reto["Mensaje_Emocional"] = info_reto["Emocional"]

    if is_alto_valor == 0 and sensibilidad_precio == 'Alta':
        if "Compra Mínima Garantizada" in reto["Tipo_Reto"] and reto["Puntos_Recompensa"] > 0:
            reto["Descripcion"] += " ¡Oferta especial por tiempo limitado!"
            reto["Puntos_Recompensa"] = int(reto["Puntos_Recompensa"] * 1.2)

    if reto["Producto_Sugerido"] == "Tu Favorito":
        if preferencia_sabor == 'Cítrico':
            reto["Producto_Sugerido"] = "Sprite o Fanta"
        elif preferencia_sabor == 'Ligero':
            reto["Producto_Sugerido"] = "Coca-Cola Zero Azúcar"
        elif preferencia_sabor == 'Intenso':
            reto["Producto_Sugerido"] = "Coca-Cola Original"
        elif preferencia_sabor == 'Frutal':
            reto["Producto_Sugerido"] = "Jugos Del Valle o Fanta"
        else:
            reto["Producto_Sugerido"] = "Coca-Cola Original"

    if momento_consumo == "Deporte":
        reto["Descripcion"] += " ¡Ideal para recargar energías después de tu entrenamiento!"
        if reto["Producto_Sugerido"] == "Variedad Coca-Cola":
            reto["Producto_Sugerido"] = "Agua Powerade o Smartwater"
    elif momento_consumo == "Social":
        reto["Descripcion"] += " ¡Perfecto para compartir con amigos y familia!"
        if reto["Producto_Sugerido"] == "Variedad Coca-Cola":
            reto["Producto_Sugerido"] = "Coca-Cola en presentaciones grandes"
    elif momento_consumo == "Descanso":
        reto["Descripcion"] += " ¡Tu acompañante ideal para esos momentos de relax!"
        if reto["Producto_Sugerido"] == "Variedad Coca-Cola":
            reto["Producto_Sugerido"] = "Coca-Cola Sin Cafeína o Agua"
    elif momento_consumo == "Trabajo":
        reto["Descripcion"] += " ¡Para mantenerte enfocado y refrescado en tu jornada laboral!"
        if reto["Producto_Sugerido"] == "Variedad Coca-Cola":
            reto["Producto_Sugerido"] = "Coca-Cola Energy o Coca-Cola Original"

    if temporada_actual == "Verano" and reto["Categoria_Enfocada"] in ["Bebidas Refrescantes", "General"]:
        reto["Descripcion"] += " ¡Refresca tu verano con Coca-Cola!"
        reto["Nombre_Reto"] += " - Reto de Verano"
        if reto["Producto_Sugerido"] == "Variedad Coca-Cola":
             reto["Producto_Sugerido"] = "Coca-Cola Original Helada"
    elif temporada_actual == "Navidad" and reto["Categoria_Enfocada"] in ["Bebidas Refrescantes", "General"]:
        reto["Descripcion"] += " ¡La magia de la Navidad se vive con Coca-Cola!"
        reto["Nombre_Reto"] += " - Reto Navideño"
        if reto["Producto_Sugerido"] == "Variedad Coca-Cola":
             reto["Producto_Sugerido"] = "Coca-Cola Edición Navideña"

    reto["Puntos_Recompensa"] = max(10, reto["Puntos_Recompensa"])

    return reto

# --- FASTAPI APP ---
app = FastAPI()

@app.get("/")
def root():
    return {"mensaje": "API de retos Coca-Cola activa"}

@app.get("/reto-cliente/{cliente_id}")
def reto_cliente(cliente_id: int):
    try:
        reto = generar_reto_para_cliente(cliente_id, df_clientes_completo, temporada_actual="Verano")
        if reto is None:
            raise HTTPException(status_code=404, detail="Cliente no encontrado o sin datos suficientes")
        return reto
    except Exception as e:
        return {"error": str(e)}

@app.get("/predecir-cliente/{cliente_id}")
def predecir_cliente(cliente_id: str):
    try:
        df = pd.read_csv("todos_los_datos.csv", encoding="latin-1")

        # Convertir cliente_id a entero
        cliente_id_int = int(cliente_id)

        cliente = df[df["ID Cliente"] == cliente_id_int]

        if cliente.empty:
            raise HTTPException(status_code=404, detail="Cliente no encontrado")

        # Aplicar encoders
        for col in encoders:
            if col in cliente.columns:
                cliente[col] = encoders[col].transform(cliente[col].astype(str))

        # Completar columnas necesarias
        for col in columnas:
            if col not in cliente.columns:
                cliente[col] = 0

        cliente_modelo = cliente[columnas]

        pred = modelo.predict(cliente_modelo)[0]
        proba = modelo.predict_proba(cliente_modelo)[0]

        reto = (
            "Compra 2 veces este mes y gana puntos dobles" if pred == 1
            else "Haz tu primera recompra y gana un cupón de $20"
        )

        return {
            "cliente_id": cliente_id_int,
            "es_cliente_alto_valor": int(pred),
            "probabilidad_bajo_valor": float(proba[0]),
            "probabilidad_alto_valor": float(proba[1]),
            "reto": reto,
        }

    except ValueError:
        # El cliente_id no es convertible a entero
        raise HTTPException(status_code=400, detail="El ID del cliente debe ser un número entero")
    except Exception as e:
        return {"error": str(e)}

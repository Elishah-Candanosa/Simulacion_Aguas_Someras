# Simulador de Aguas Someras 2D 

Un simulador numérico para la dinámica de fluidos que resuelve las Ecuaciones de Aguas Someras en dos dimensiones. Este código modela la evolución del flujo alrededor de un islote topográfico, calculando las fuerzas hidrodinámicas y diagnosticando el régimen del flujo.

<img width="2816" height="1369" alt="Islote_Demon" src="https://github.com/user-attachments/assets/77139f4d-9234-4356-88d2-0590ff4b6856" />

## Demostración Visual

A continuación se muestra la simulación que el código genera tal cual está en este repositorio. Tenga en consideración que esta simulación se aceleró para su mejor visualización, en el código la animación va **bastante** más lento.

https://github.com/user-attachments/assets/f42a8971-5cc0-4842-ac14-fed41b4b5a75





*(Espacio reservado para video secundario o gráficas)*

## Física y Métodos Numéricos

El simulador resuelve el sistema hiperbólico de conservación de masa y momento bidimensional en presencia de una topografía de fondo $z_b$:

$$U_t + \nabla \cdot F(U) = S(U)$$

Donde el vector de variables conservadas es $U = [h, hu, hv]^T$. Para garantizar la estabilidad y precisión matemática en la formación de ondas y frentes de choque, la implementación utiliza:

* **Esquema de Flujo:** Flujo de Rusanov.
* **Resolución Espacial:** Reconstrucción MUSCL de 2º orden acoplada con un limitador de pendiente *minmod* para evitar oscilaciones espurias.
* **Integración Temporal:** Método Runge-Kutta de Variación Total Decreciente de 2 etapas (TVD-RK2).
* **Fronteras:** Condiciones de contorno reflectantes.

## Diagnósticos Integrados

Durante y al finalizar la ejecución, el script genera figuras que evalúan:
* Conservación de masa y evolución de la energía mecánica total.
* Magnitud de las fuerzas hidrodinámicas (componentes dinámicas e hidrostáticas) ejercidas sobre el obstáculo sólido.
* Evolución temporal del número de Froude máximo en el dominio para identificar transiciones de régimen subcrítico a supercrítico.

## Requisitos y Ejecución

El proyecto está escrito en Python. Así que es prudente que en su sistema alguna versión de Python se encuentre instalado


## Contribuidores y contacto:
* Carlos Elishah Candanosa Salazar: Discretización de ecuaciones e implementación del Código. En caso de tener alguna sugerencia o pregunta, ¡no duden en contactarme a mi correo institucional, carlos.candanosa@correo.nucleares.unam.mx!
  
* Carlos Alejandro Ávila Hernández: Modelo Teórico y Documentación.


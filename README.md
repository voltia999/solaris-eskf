# ESKF — Error-State Kalman Filter

Implementación en Python de un filtro de Kalman de estado de error (ESKF) de 15 estados para navegación inercial, con cuaterniones de Hamilton.

**Estado nominal:** posición `p`, velocidad `v`, orientación `q`, sesgo del acelerómetro `b_a` y sesgo del giróscopo `b_g`.

**Actualizaciones disponibles:** GPS (posición y velocidad), magnetómetro, acelerómetro (gravedad) y barómetro (altura).

La formulación matemática completa está en [`docs/ESKF.pdf`](docs/ESKF.pdf).

## Estructura

```
eskf.py            # Clase ESKF: predicción, actualizaciones y run()
sensors.py         # Sensor: última medida de cada sensor, consumida por run()
quaternion.py      # Operaciones con cuaterniones y jacobianos
linalg_utils.py    # Utilidades de álgebra lineal (matriz antisimétrica)
test/              # Tests con pytest
docs/ESKF.pdf      # Documento con la derivación
docs/revision_ESKF # Revisión del documento (errores y comprobaciones)
```

## Instalación

```bash
pip install -r requirements.txt        # solo numpy
pip install -r requirements-dev.txt    # + pytest y scipy para los tests
```

## Uso

```python
import numpy as np
from eskf import ESKF
from sensors import Sensor

f = ESKF(sigma_a=0.05, sigma_w=0.005, sigma_aw=1e-4, sigma_ww=1e-5,
         mag_ref=np.array([0.21, 0.0, 0.43]))  # campo magnético en NED

# Un Sensor por fuente; new_meas() al llegar cada lectura.
sensors = [Sensor("accel", ts=-np.inf, data=None, sigma=0.1, last_ts=-np.inf),
           Sensor("mag",   ts=-np.inf, data=None, sigma=0.01, last_ts=-np.inf),
           Sensor("baro",  ts=-np.inf, data=None, sigma=0.5, last_ts=-np.inf),
           Sensor("gps",   ts=-np.inf, data=None, sigma=(2.0, 0.1), last_ts=-np.inf)]

# En cada muestra de la IMU: predicción y una única actualización con las medidas
# nuevas (sec. 4.5.3). El magnetómetro solo se usa si se acepta el acelerómetro.
sensors[3].new_meas(t, np.concatenate([p_gps, v_gps]))   # GPS: [p; v] en NED
used = f.run(a_meas, w_meas, dt, sensors)                # p. ej. ["accel", "mag", "gps"]

# También se puede aplicar una medida suelta:
f.update_gps(p_meas, v_meas, sigma_p=2.0, sigma_v=0.1)
f.update_mag(m_meas, sigma_mag=0.01)
f.update_accel(a_meas, sigma_a=0.1)
f.update_baro(z_meas, sigma_baro=0.5)

print(f.p, f.v, f.q)
```

## Tests

```bash
pytest
```

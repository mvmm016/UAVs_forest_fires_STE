#Ela funciona tanto com escalares (o mundo real) quanto com arrays (as N partículas), graças ao NumPy.
import numpy as np
# Calcula a concentração de gás num ponto do espaço, dada uma fonte.
def gaussian_plume_gp(sensor_pos, source_pos, qs, us, zetas, wind_euler):
    """
    Gaussian Plume (GP) model using Euler angles for wind direction.
    """
    X, Y, sensor_height = sensor_pos
    xs, ys, zs = source_pos
    zeta1, zeta2 = zetas

    pitch, yaw = wind_euler

    # Conversão de Ângulos de Euler (Pitch e Yaw) para Vetor Unitário 3D.
    # Assumindo a convenção em que Yaw é a rotação no plano XY e Pitch é a elevação.

    #Converte os ângulos do vento (pitch, yaw) num vetor unitário 3D (ux, uy, uz).
    ux = np.cos(yaw) * np.cos(pitch)
    uy = np.sin(yaw) * np.cos(pitch)
    uz = np.sin(pitch)

#Calcula a diferença entre sensor e fonte (dx, dy, dz).
    dx = X - xs
    dy = Y - ys
    dz = sensor_height - zs

#Decompõe essa diferença em:
    #dk: distância na direção do vento (a favor/contra).
    #ck: distância transversal ao vento.
    dk = dx * ux + dy * uy + dz * uz
    ck = -dx * uy + dy * ux

#sigma_y e sigma_z são as dispersões da pluma, crescendo com a distância r (a fórmula usa zeta1/zeta2 como coeficientes de difusão).
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r_safe = np.maximum(r, 1e-10)

    sigma_y = (zeta1 * r_safe) / np.sqrt(1 + 0.0001 * r_safe)
    sigma_z = (zeta2 * r_safe) / np.sqrt(1 + 0.0001 * r_safe)

#A concentração é o produto de três termos:
    #scale_term = q / (u·σy·σz·2π): intensidade geral.
    #cross_term: gaussiana no eixo transversal.
    #vert_term: gaussiana vertical mais um termo espelhado (sensor_height + zs), que modela a reflexão da pluma no solo.
    scale_term = qs / (us * sigma_y * sigma_z * 2 * np.pi)
    cross_term = np.exp(-(ck**2) / (2 * sigma_y**2))

    vert_term = np.exp(-(dz**2) / (2 * sigma_z**2)) + \
                np.exp(-((sensor_height + zs)**2) / (2 * sigma_z**2))

    concentration = scale_term * cross_term * vert_term
    #np.where(dk > 0, ..., 0) zera a concentração contra o vento (atrás da fonte).
    concentration = np.where(dk > 0, concentration, 0.0)

    return concentration
"""Comando anterior de validación del MLP; delega en los chequeos comunes."""
from tps_sia.tp3.shared.tests.test_mlp import (
    X_XOR, Y_XOR, comprobar_gradientes, main, test_entradas_invalidas,
    test_gradient_check, test_guardar_cargar_y_continuar,
    test_historia_y_validacion, test_inicializacion, test_sgd_y_mini_batches,
    test_softmax_estable, test_xor,
)


if __name__ == "__main__":
    main()

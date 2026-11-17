
import numpy as np
import pandas as pd


def pandas_derivata_robusta(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hampel_window: int = 9,
    n_sigma: float = 3.0,
    smooth_window: int = 9,
    degree: int = 2    
    ) -> pd.DataFrame:
    """
    Pipeline completa per la pulizia e la derivazione numerica
    di un segnale rappresentato in una tabella pandas.

    Data la tabella, df, in cui i valori campionati sono nella colonna, y_col
    rispetto ai valori della colonna, x_col, la funzione restituisce una nuova tabella:
        - con i dati originali ordinati secondo x_col
        - una colonna con gli 'aberranti'
        - una con 'misura_clean' senza aberranti
        - una con 'misura_smooth', ovvero un y_col lisciata
        - una con la 'derivata' rispetto a x_col 
    
    x_col, y_col: nomi delle colonne
    hampel_window, n_sigma: finestra dei valori di rimozione degli aberranti con Hampel ed estensione della deviazione ammessa
    smooth_window, degree: finestra dei punti della regressione polinomiale locale e grado del polinomio
    
    Passaggi del calcolo:
      1. Ordinamento rispetto a x.
      2. Rilevamento degli aberranti con filtro di Hampel.
      3. Rimozione degli aberranti.
      4. Interpolazione lineare dei valori rimossi.
      5. Regressione polinomiale locale.
      6. Calcolo della derivata rispetto a x.

    Le derivate agli estremi sono lasciate a NaN perché non è
    disponibile una finestra simmetrica completa.

    """

    def filtro_hampel(
        s: pd.Series,
        window: int = 9,
        n_sigma: float = 3.0
        ) -> pd.Series:
            """
            Identifica i valori aberranti in s mediante un filtro di Hampel centrato.

            La funzione data la serie pandas, s, restituisce una nuova serie pandas
            con tutti valori booleani, dove VERO corrisponde ad un valore aberrante
            e FALSO ad un valore non aberrante.

            Il criterio di decisione sull'aberranza è dato da una deviazione del valore
            maggiore di n_sigma volte il valore della deviazione robusta rispetto ai
            valori vicini compresi nella finestra (window).

            La deviazione robusta è identificata con la MAD, median absolute deviation.
            La deviazione assoluta mediana, abbreviata MAD, è una misura di dispersione 
            basata sulla mediana. Si calcola in tre passaggi:
            -   si trova la mediana dei dati;
            -   si calcola la distanza assoluta di ogni valore da quella mediana;
            -   si prende la mediana di queste distanze.
            In formula: MAD = mediana(|xᵢ - mediana(x)|)

            Le osservazioni per cui non è disponibile una finestra completa
            non vengono classificate come aberranti.

            """
            if window < 3 or window % 2 == 0:
                raise ValueError("window deve essere dispari e >= 3.")

            if not np.isfinite(n_sigma) or n_sigma <= 0:
                raise ValueError("n_sigma deve essere positivo e finito.")

            if not np.all(np.isfinite(s.to_numpy(dtype=float))):
                raise ValueError("La serie contiene valori NaN o inf.")

            # calcola una nuova serie pandas mediana ristretta per ciascun valore in s sulla sua finestra (window)
            mediana = s.rolling(
                window=window,
                center=True,
                min_periods=window # per gli estremi non in finestra assegna il valore NaN
                ).median()

            # calcola una nuova serie pandas MAD di finestra per ciascun valore in s
            mad = s.rolling(
                window=window,
                center=True,
                min_periods=window
                    ).apply(lambda values: np.median(np.abs(values - np.median(values))),
                            raw=True)

            robust_sigma = 1.4826 * mad
            deviation = (s - mediana).abs()

            # Le finestre incomplete producono NaN e non classificano
            # l'osservazione come aberrante.
            aberranti = deviation > n_sigma * robust_sigma

            return aberranti.fillna(False).astype(bool)

    def liscia_deriva_polinomiale(
        x: np.ndarray,
        y: np.ndarray,
        window: int = 9,
        degree: int = 2
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dato il campionamento nel numpy.array y in relazione ai corrispondenti valori nel numpy.array x,
        la funzione liscia_deriva_polinomiale() fornisce una tupla contenente:
        - l'array con i valori di y lisciati
        - la loro derivata rispetto a x
        usando un polinomiio locale di ordine degree su un numero di punti circostanti pari a window.

        La regressione polinomiale viene eseguita su finestre simmetriche e complete.
        I primi e gli ultimi (window - 1) // 2 campioni restituiscono NaN.

        La variabile x può essere non equispaziata.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError(
                "x e y devono essere array monodimensionali della stessa lunghezza."
            )

        if window < 3 or window % 2 == 0:
            raise ValueError("window deve essere dispari e >= 3.")

        if degree < 1 or degree >= window:
            raise ValueError("degree deve essere >= 1 e < window.")

        if len(x) < window:
            raise ValueError("Numero di campioni inferiore alla finestra.")

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("x e y devono contenere valori finiti.")

        if not np.all(np.diff(x) > 0):
            raise ValueError("x deve essere strettamente crescente.")

        n = len(x)
        half = window // 2

        smooth = np.full(n, np.nan)
        derivative = np.full(n, np.nan)

        for i in range(half, n - half):

            # Finestra simmetrica e completa
            idx = np.arange(i - half, i + half + 1)

            # Coordinate centrate nel punto in cui si calcola
            # la derivata.
            dx = x[idx] - x[i]

            # Normalizzazione delle coordinate per migliorare
            # il condizionamento numerico del fit.
            scale = np.max(np.abs(dx))
            z = dx / scale

            # Fit polinomiale locale
            coeff = np.polyfit(z, y[idx], degree)

            # Valore stimato nel centro della finestra
            smooth[i] = np.polyval(coeff, 0.0)

            # Derivata rispetto a x, non rispetto a z
            derivative[i] = coeff[-2] / scale

        return smooth, derivative
    
    # Verifica delle colonne
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Le colonne specificate non esistono nel DataFrame.")

    if x_col == y_col:
        raise ValueError("x_col e y_col devono essere colonne diverse.")

    # Verifica dei parametri
    if hampel_window < 3 or hampel_window % 2 == 0:
        raise ValueError("hampel_window deve essere dispari e >= 3.")

    if smooth_window < 3 or smooth_window % 2 == 0:
        raise ValueError("smooth_window deve essere dispari e >= 3.")

    if degree < 1 or degree >= smooth_window:
        raise ValueError("degree deve essere >= 1 e < smooth_window.")

    if not np.isfinite(n_sigma) or n_sigma <= 0:
        raise ValueError("n_sigma deve essere positivo e finito.")

    # Copia e ordinamento
    result = df.copy().sort_values(x_col, kind="stable")

    # Conversione delle colonne in array numerici
    try:
        x = pd.to_numeric(result[x_col], errors="raise").to_numpy(dtype=float)
        y = pd.to_numeric(result[y_col], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Le colonne x e y devono contenere dati numerici.") from exc

    # Controlli sui dati
    if len(result) < max(hampel_window, smooth_window):
        raise ValueError("Numero di campioni insufficiente.")

    if not np.all(np.isfinite(x)):
        raise ValueError("La variabile indipendente contiene NaN o inf.")

    if not np.all(np.isfinite(y)):
        raise ValueError("La variabile dipendente contiene NaN o inf.")

    if not np.all(np.diff(x) > 0):
        raise ValueError(
            "La variabile indipendente deve essere strettamente crescente "
            "e non contenere duplicati."
        )

    # 1. Rilevamento degli outlier
    aberranti = filtro_hampel(
        pd.Series(y),
        window=hampel_window,
        n_sigma=n_sigma
    ).to_numpy()

    result["aberranti"] = aberranti

    # 2. Rimozione degli outlier
    y_clean = y.copy()
    y_clean[aberranti] = np.nan

    # 3. Interpolazione lineare usando i valori effettivi di x
    y_clean = (
        pd.Series(y_clean, index=x)
        .interpolate(method="index", limit_area="inside")
        .to_numpy()
    )

    if not np.all(np.isfinite(y_clean)):
        raise ValueError(
            "Non è possibile interpolare tutti i valori mancanti. "
            "Verificare la distribuzione degli outlier."
        )

    result["misura_clean"] = y_clean

    # 4. Fit polinomiale locale e derivata
    smooth, derivative = liscia_deriva_polinomiale(
        x=x,
        y=y_clean,
        window=smooth_window,
        degree=degree
    )

    result["misura_smooth"] = smooth
    result["derivata"] = derivative

    return result

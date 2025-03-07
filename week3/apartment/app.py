import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Modell laden
model_filename = "random_forest_regression.pkl"
with open(model_filename, "rb") as f:
    model = pickle.load(f)

# CSV-Daten laden
df_bfs_data = pd.read_csv("bfs_municipality_and_tax_data.csv", sep=",", encoding="utf-8")
df_bfs_data["tax_income"] = df_bfs_data["tax_income"].str.replace("'", "").astype(float)

# Neue Features berechnen
df_bfs_data["emp_per_capita"] = df_bfs_data["emp"] / df_bfs_data["pop"]
df_bfs_data["wealth_factor"] = df_bfs_data["tax_income"] * df_bfs_data["emp_per_capita"]

# Alle Ortschaften
locations = {
    "Zürich": 261, "Kloten": 62, "Uster": 198, "Illnau-Effretikon": 296, "Feuerthalen": 27,
    "Pfäffikon": 177, "Ottenbach": 11, "Dübendorf": 191, "Richterswil": 138, "Maur": 195,
    "Embrach": 56, "Bülach": 53, "Winterthur": 230, "Oetwil am See": 157, "Russikon": 178,
    "Obfelden": 10, "Wald (ZH)": 120, "Niederweningen": 91, "Dällikon": 84, "Buchs (ZH)": 83,
    "Rüti (ZH)": 118, "Hittnau": 173, "Bassersdorf": 52, "Glattfelden": 58, "Opfikon": 66,
    "Hinwil": 117, "Regensberg": 95, "Langnau am Albis": 136, "Dietikon": 243, "Erlenbach (ZH)": 151,
    "Kappel am Albis": 6, "Stäfa": 158, "Zell (ZH)": 231, "Turbenthal": 228, "Oberglatt": 92,
    "Winkel": 72, "Volketswil": 199, "Kilchberg (ZH)": 135, "Wetzikon (ZH)": 121, "Zumikon": 160,
    "Weisslingen": 180, "Elsau": 219, "Hettlingen": 221, "Rüschlikon": 139, "Stallikon": 13,
    "Dielsdorf": 86, "Wallisellen": 69, "Dietlikon": 54, "Meilen": 156, "Wangen-Brüttisellen": 200,
    "Flaach": 28, "Regensdorf": 96, "Niederhasli": 90, "Bauma": 297, "Aesch (ZH)": 241,
    "Schlieren": 247, "Dürnten": 113, "Unterengstringen": 249, "Gossau (ZH)": 115,
    "Oberengstringen": 245, "Schleinikon": 98, "Aeugst am Albis": 1, "Rheinau": 38, "Höri": 60,
    "Rickenbach (ZH)": 225, "Rafz": 67, "Adliswil": 131, "Zollikon": 161, "Urdorf": 250,
    "Hombrechtikon": 153, "Birmensdorf (ZH)": 242, "Fehraltorf": 172, "Weiach": 102,
    "Männedorf": 155, "Küsnacht (ZH)": 154, "Hausen am Albis": 4, "Hochfelden": 59,
    "Fällanden": 193, "Greifensee": 194, "Mönchaltorf": 196, "Dägerlen": 214, "Thalheim an der Thur": 39,
    "Uetikon am See": 159, "Seuzach": 227, "Uitikon": 248, "Affoltern am Albis": 2, "Geroldswil": 244,
    "Niederglatt": 89, "Thalwil": 141, "Rorbas": 68, "Pfungen": 224, "Weiningen (ZH)": 251,
    "Bubikon": 112, "Neftenbach": 223, "Mettmenstetten": 9, "Otelfingen": 94, "Flurlingen": 29,
    "Stadel": 100, "Grüningen": 116, "Henggart": 31, "Dachsen": 25, "Bonstetten": 3,
    "Bachenbülach": 51, "Horgen": 295
}

# Vorhersagefunktion
def predict(rooms, area, town):
    if town not in locations:
        return "Stadt nicht gefunden!"
    
    bfs_number = locations[town]
    df = df_bfs_data[df_bfs_data["bfs_number"] == bfs_number].copy()
    df.reset_index(inplace=True)

    if len(df) != 1:
        return "Keine eindeutigen Daten für diese Stadt!"

    # Benutzerwerte (`rooms` und `area`) hinzufügen
    df.loc[0, "rooms"] = rooms
    df.loc[0, "area"] = area

    # Nur die Features verwenden, mit denen das Modell trainiert wurde
    prediction = model.predict(df[['pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income', 'emp_per_capita', 'wealth_factor']])

    return f"Vorhergesagter Mietpreis: {round(prediction[0], 2)} CHF"

# Gradio-Interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Anzahl Zimmer"),
        gr.Number(label="Wohnfläche (m²)"),
        gr.Dropdown(choices=locations.keys(), label="Stadt"),
    ],
    outputs="text",
    title="Mietpreis-Vorhersage",
    description="Gibt eine Mietpreis-Vorhersage für eine Wohnung basierend auf Stadt, Wohnfläche und Zimmeranzahl aus.",
    examples=[
        [4.5, 120, "Dietlikon"],
        [3.5, 60, "Winterthur"],
        [2, 40, "Zürich"],
    ],
)

# App starten
if __name__ == "__main__":
    demo.launch()

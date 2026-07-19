import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Convert time columns
    df["Arrival_time"] = pd.to_datetime(
        df["Arrival_time"], format="%H:%M:%S", errors="coerce"
    )

    df["Departure_Time"] = pd.to_datetime(
        df["Departure_Time"], format="%H:%M:%S", errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna()

    return df


def create_train_dataset(df):
    train_rows = []

    for train_no, group in df.groupby("Train_No"):

        group = group.sort_values("Distance")

        start_station = group.iloc[0]["Station_Name"]
        end_station = group.iloc[-1]["Station_Name"]

        total_distance = group["Distance"].max()

        number_of_stops = len(group)

        departure = group.iloc[0]["Departure_Time"]
        arrival = group.iloc[-1]["Arrival_time"]

        duration = (arrival - departure).total_seconds() / 60

        if duration < 0:
            duration += 24 * 60

        train_rows.append({
            "Train_No": train_no,
            "Start_Station": start_station,
            "End_Station": end_station,
            "Distance": total_distance,
            "Stops": number_of_stops,
            "Journey_Duration": duration
        })

    return pd.DataFrame(train_rows)
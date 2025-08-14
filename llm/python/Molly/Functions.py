import matplotlib.pyplot as plt
import contextily as ctx
import re
import folium
import geopandas as gpd
import branca.colormap as cm
import numpy as np
from scipy.stats import zscore, gaussian_kde
import os
from datetime import datetime
import pandas as pd
import os
import webbrowser


def make_scatter_plot(df, column):
    df_notna = df[df['gentrification_prediction'].notna()].copy()
    df_notna = df[df[column].notna()].copy()
    # Convert columns to numpy arrays
    x = df_notna['gentrification_prediction'].to_numpy()
    y = df_notna[column].to_numpy()
    
    # Compute point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    # Sort points by density for better visualization
    idx = np.argsort(z)  # Ensure this is properly applied to NumPy arrays
    x, y, z = x[idx], y[idx], z[idx]
    
    # Create scatter plot with density-based coloring
    plt.figure(figsize=(4, 3))
    sc = plt.scatter(x, y, c=z, cmap='viridis', edgecolors='none', alpha=0.7)
    # plt.colorbar(sc, label="Density")
    plt.xlabel("Gentrification Prediction")
    plt.ylabel(f"{column}");

def add_relative_price(df):
    # Clean and convert 'price' column to float, handling commas and dollar signs
    df['price_float'] = \
    (
        df['price']
        .replace({None: np.nan})           # Replace None with NaN
        .str.replace(r'[\$,]', '', regex=True)  # Remove $ and , using regex
        .astype(float)                     # Convert to float
    )
    
    # Need a z score because there is a massive price
    df['price_z'] = zscore(df['price_float'], nan_policy='omit')
    
    # Even after normalising a couple of huge prices still skews loads, get rid of them
    df.loc[df.price_z > 2, 'price'] = np.nan
    df.loc[df.price_z > 2, 'price_z'] = np.nan
    
    # Finally take account of the property size (bigger -> more expensive)
    df['price_relative'] = df['price_z'] / df['accommodates']
    return df

def map_not_static(neighbourhoods, df, scoring_col, exclude_nans=False, open_in_browser=False):
    if exclude_nans:
        df = df[df[scoring_col] != 'Na']

    # print(np.unique(df[scoring_col]))
    df_notna = df.copy()  # No filtering of NaN values

    # Get map center
    map_center = [df_notna["latitude"].mean(), df_notna["longitude"].mean()]

    # Create a Folium map
    m = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB Positron")

    # Add neighborhoods as GeoJSON
    folium.GeoJson(
        neighbourhoods,
        name="Neighborhoods",
        style_function=lambda feature: {
            "fillColor": "none",
            "color": "black",
            "weight": 0.5,
        },
    ).add_to(m)

    # Define a fixed color mapping for scoring values
    color_mapping = {
        1: "lightgray",
        2: "indianred",
        3: "darkorange",
        4: "darkred",
        0: "cyan"
    }

    # Define a color legend using StepColormap
    colormap = cm.StepColormap(
        colors=[color_mapping[key] for key in sorted(color_mapping.keys())],
        vmin=min(color_mapping.keys()),
        vmax=max(color_mapping.keys()),
        index=list(color_mapping.keys()),
        caption="Gentrification Score",
    )
    
    m.add_child(colormap)

    # Add listings as circle markers
    for _, row in df_notna.iterrows():
        popup_text = (
            f"Text: {row['text']}<br>"
            f"Score_G: {row['gentrification_prediction']}<br>"
            f"Score_N: {row['niceness_score']}<br>"
            f"Score_G_num: {row['gentrification_num_score']}<br>"
            f"Explanation_G: {row['explanation']}<br>"
            f"Explanation_N: {row['niceness_explanation']}<br>"
            f"Explanation_G_num: {row['gentrification_num_explanation']}"
        
        )

        # Assign color based on predefined mapping or default to gray
        color = "lime" if pd.isna(row[scoring_col]) else color_mapping.get(row[scoring_col], "gray")

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=300),
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    if open_in_browser:
        m.save("map.html")
        map_path = os.path.abspath("map.html")
        webbrowser.open(map_path)

    return m

def map_static(neighbourhoods, df, column = 'id', save_fig = False):
    
    # Create a figure with the desired size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    neighbourhoods.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.3, zorder=3)
    
    # Plot the listings, colouring by gentrification score
    if column == 'id':
        df.plot(ax=ax, column=column, markersize=1.5, color='red', alpha=1, zorder=2, )
    else:
        df.plot(ax=ax, column=column, markersize=1.5, alpha=0.8, zorder=2, legend=True)
    
    # Add a basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=neighbourhoods.crs.to_string(), zorder=1)
    
    # Adjust the axis for a clean appearance
    ax.set_axis_off()

    if save_fig == True:
        ax.figure.savefig("airbnb-bristol/listing_scores.png", dpi=300, bbox_inches="tight")


# Create a log with the current time
LOG_FILE = os.path.join("../logs", datetime.now().strftime("%Y-%m-%d-%H%M%S.log"))
def log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg)

def get_gentrification_scores_categorical_one_per_lsoa(batch_tweets, system_prompt, client, batch_index=0, max_tokens=1000):
    """
    Retrieves gentrification scores for a batch of tweets using the Together AI API.

    Returns:
    -------
    ids : list of int
        DataFrame indices for each tweet.
    sentiments : list of str
        Gentrification categories: 'Established', 'Gentrifying', 'Emerging', 'Undeveloped', or np.nan for 'NA'.
    explanations : list of str
        LLM-provided reasoning behind each score.
    """

    import re
    import numpy as np
    from datetime import datetime

    # Prepare the list of tweets
    tweet_list = "\n".join([f"{idx + 1}. {tweet}" for idx, tweet in enumerate(batch_tweets.text.values)])
    system_prompt = f"{system_prompt}\n\n{tweet_list}"
    # print(f"{len(batch_tweets)} listings in listing list")
    
    # Prepare API messages
    messages = [{"role": "system", "content": system_prompt}]

    # API Call
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1,
        stream=False
    )

    # Extract the assistant's reply
    assistant_reply = response.choices[0].message.content.strip()

    # Log for debugging
    log(f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}\n"
        f"**MESSAGE**\n{messages}\n"
        f"**RESPONSE**\n{assistant_reply}\n\n")

    # Clean response
    assistant_reply = re.sub(r'^\s*Here are the (scores|analyses):\s*', '', assistant_reply, flags=re.IGNORECASE).strip()

    # Updated regex pattern for the new format
    pattern = r'''
        ^\s*                                # Start of line, allow leading whitespace
        (\d+)                               # Capture Group 1: Line number (digits only)
        \s*[.:]\s*                          # Dot or colon separator after the number
        (Established|Gentrifying|Emerging|Undeveloped|NA)  # Group 2: Categorical score
        [.:]\s*                             # Dot or colon after the score
        (.*)                                # Group 3: Explanation (rest of the line)
        $                                   # End of the line
    '''

    # Parse the response
    ids = []
    scores = []
    explanations = []
    error_count = 0

    lines = assistant_reply.split('\n')
    for i, line in enumerate(lines):
        # print(f"this is i : {i}") 
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line, re.VERBOSE | re.IGNORECASE)
        if match:
            index = int(match.group(1))
            score = match.group(2).capitalize()
            explanation = match.group(3).strip()

            log(f"{i} {line}\n\t{index},{score},{explanation}")
            # print(f"{i} {line}\n\t{index},{score},{explanation}")
            
            # Validate score
            valid_scores = ["Established", "Gentrifying", "Emerging", "Undeveloped", 'Na']
            if score in valid_scores:
                ids.append(index)
                scores.append(np.nan if score == "NA" else score)
                explanations.append(explanation)
            else:
                msg = f"Warning: Invalid score '{score}' on line {i}: '{line}'"
                print(msg)
                log(msg)
                error_count += 1
        else:
            msg = f"\n*********************\n" \
                  f"Warning: Invalid format on line {i}: '{line}'.\n" \
                  f"The full response was: \n{assistant_reply}\n" \
                  f"*********************\n"
            print(msg)
            log(msg)
            error_count += 1
            break

        if index - 1 >= len(batch_tweets):
            msg = f"Found {index} tweets, but there are more lines. Assuming remaining lines are junk and ignoring them."
            log(msg)
            print(msg)
            break

    if error_count > 0:
        scores = [np.nan] * len(batch_tweets)
        ids = [x + 1 for x in range(len(batch_tweets))]
        explanations = ["ERROR"] * len(batch_tweets)

    # Compute DataFrame indices
    df_ids = [batch_index + int(id) - 1 for id in ids]

    assert len(df_ids) == len(scores), f"Length of ids ({len(df_ids)}) does not match length of scores ({len(scores)})."

    return df_ids, scores, explanations

def get_gentrification_scores(batch_tweets, system_prompt, client, batch_index=0, max_tokens=1000):
    """
    Retrieves gentrification scores for a batch of tweets using the Together AI API.

    Parameters
    ----------
    batch_tweets : pandas.DataFrame
        A DataFrame containing the tweets for the current batch.
        It must include a 'text' column with the tweet content.
    system_prompt : str
        The system prompt to be sent to the Together AI API.
        The tweet texts will be appended to this prompt.
    client : Together
        The Together client object to use for API calls.
    batch_index : int
        An optional starting index of the current batch.
        This is used to align the predicted sentiments with the original DataFrame indices.

    Returns
    -------
    ids : list of int
        A list of DataFrame indices corresponding to each tweet in the batch.
        These indices align with the main DataFrame from which this batch was drawn.
    sentiments : list of str
        A list of predicted gentrification scores for each tweet in the batch.
        Possible values are 1 (not suggestive of gentrification) to 5 (highly suggestive).
    explanations : list of str
        Explanations that the LLM returns giving it's reason for the the chosen score
        (these may or may not happen depending on the prompt, and the LLM's mood!)
    """

    # Prepare the list of tweets
    tweet_list = "\n".join([f"{idx + 1}. {tweet}"
                            for idx, tweet in enumerate(batch_tweets.text.values)])

    # Create the system prompt
    system_prompt = f"{system_prompt}\n\n{tweet_list}"
    #print("PROMPT:", system_prompt, "\n\n")

    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Call the API using parameters that ChatGPT recommends for this task
    response = client.chat.completions.create(
        #model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=messages,
        max_tokens=max_tokens,  # max length of output (in case I get the prompt wront and it talks for ages...)
        temperature=0.2,  # lower for more deterministic
        top_p=0.9,  # ??
        top_k=40,  # ??
        repetition_penalty=1,
        stop=["""",
              """],
        #truncate=130560,  # ??
        stream=False  # Set stream to False to get the full response
    )

    # Extract the assistant's reply and get the IDs and scores
    assistant_reply = response.choices[0].message.content.strip()

    # Useful to have a full log for debugging etc
    log(f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}\n" \
        f"**MESSAGE**\n{messages}\n" \
        f"**RESPONSE**\n{assistant_reply}\n\n")

    # Parse the IDs and scores (and, if available the explanation for the score)
    ids = []
    scores = []
    explanations = []

    # Regex pattern to extract the line number, score, and optional text from a line
    pattern = r'''
        ^\s*              # Start of line, allow leading whitespace
        (\d+)             # Capture Group 1: The line number
        \s*[.:]\s*        # A dot or colon with optional whitespace
        (?:Score:\s+)?    # Optionally match "Score:" followed by one or more spaces
        ([\d]+|NA)        # Capture Group 2: The score (one or more digits or 'NA')
        (?:\s*(.*))?      # Optional text after the score (Group 3)
        $                 # End of the line
    '''

    # Desipte being told not to, sometimes the reply starts with
    # 'Here are the scores:' or 'Here are the analyses:'
    # Remove that, and any whitespace at the start or end
    assistant_reply = re.sub(r'^\s*Here are the scores:\s*', '', assistant_reply).strip()
    assistant_reply = re.sub(r'^\s*Here are the analyses:\s*', '', assistant_reply).strip()

    # Analyse the reply line-by-line
    lines = assistant_reply.strip().split('\n')
    error_count = 0  # Return -1 on an error (and count the number of errors at the same time)
    for i, line in enumerate(lines):
        # Ignore lines that are empty once they have been stripped
        line = line.strip()
        if not line:
            continue

        # Try to match the line
        match = re.match(pattern, line, re.VERBOSE)
        if match:
            # Extract the index and score from the match groups
            index = int(match.group(1))
            score = match.group(2)
            explanation = match.group(3)
            log(f"{i} {line}\n\t{index},{score},{explanation}")
            # Validate the score range
            if score == 'NA':
                ids.append(index)
                scores.append(np.nan)  # Use np.nan to represent missing values
                explanations.append(explanation)
            elif 1 <= int(score) <= 5:
                ids.append(index)
                scores.append(int(score))
                explanations.append(explanation)
            else:
                msg = f"Warning: Score {score} out of range on line {i}: '{line}'"
                print(msg)
                log(msg)
                error_count += 1
        else:
            msg = f"\n*********************\n" \
                  f"Warning: Invalid format on line {i}: '{line}'.\n" \
                  f"The full response was: \n{assistant_reply}\n" \
                  f"*********************\n"
            print(msg)
            log(msg)
            error_count += 1
            break

        if index-1 >= len(batch_tweets):
            msg = f"Found {index} tweets, but there are more lines. Assuming remaining lines are junk and ignoring them."
            log(msg)
            print(msg)
            break

    if error_count > 0:
        # There was an error, set scores to np.nan and assume tweet IDs from
        # 1 to len(batch_tweets).
        scores = [np.nan] * len(batch_tweets)
        ids = [x + 1 for x in range(len(batch_tweets))]
        explanations = ["ERROR"] * len(batch_tweets)

    # Compute dataframe indices
    df_ids = [batch_index + int(id) - 1 for id in ids]

    assert len(df_ids) == len(scores), f"Length of ids ({len(df_ids)} does not match length of scores ({len(scores)})."
    
    return df_ids, scores, explanations

def get_gentrification_scores_categorical(batch_tweets, system_prompt, client, batch_index=0, max_tokens=1000):
    """
    Retrieves gentrification scores for a batch of tweets using the Together AI API.

    Returns:
    -------
    ids : list of int
        DataFrame indices for each tweet.
    sentiments : list of str
        Gentrification categories: 'Established', 'Gentrifying', 'Emerging', 'Undeveloped', or np.nan for 'NA'.
    explanations : list of str
        LLM-provided reasoning behind each score.
    """

    import re
    import numpy as np
    from datetime import datetime

    # Prepare the list of tweets
    tweet_list = "\n".join([f"{idx + 1}. {tweet}" for idx, tweet in enumerate(batch_tweets.text.values)])
    system_prompt = f"{system_prompt}\n\n{tweet_list}"

    # Prepare API messages
    messages = [{"role": "system", "content": system_prompt}]

    # API Call
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1,
        stream=False
    )

    # Extract the assistant's reply
    assistant_reply = response.choices[0].message.content.strip()

    # Log for debugging
    log(f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}\n"
        f"**MESSAGE**\n{messages}\n"
        f"**RESPONSE**\n{assistant_reply}\n\n")

    # Clean response
    assistant_reply = re.sub(r'^\s*Here are the (scores|analyses):\s*', '', assistant_reply, flags=re.IGNORECASE).strip()

    # Updated regex pattern for the new format
    pattern = r'''
        ^\s*                                # Start of line, allow leading whitespace
        (\d+)                               # Capture Group 1: Line number (digits only)
        \s*[.:]\s*                          # Dot or colon separator after the number
        (Established|Gentrifying|Emerging|Undeveloped|NA)  # Group 2: Categorical score
        [.:]\s*                             # Dot or colon after the score
        (.*)                                # Group 3: Explanation (rest of the line)
        $                                   # End of the line
    '''

    # Parse the response
    ids = []
    scores = []
    explanations = []
    error_count = 0

    lines = assistant_reply.split('\n')
    for i, line in enumerate(lines):
        print(line)
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line, re.VERBOSE | re.IGNORECASE)
        if match:
            index = int(match.group(1))
            print(index)
            score = match.group(2).capitalize()
            explanation = match.group(3).strip()

            log(f"{i} {line}\n\t{index},{score},{explanation}")

            # Validate score
            valid_scores = ["Established", "Gentrifying", "Emerging", "Undeveloped", 'Na']
            if score in valid_scores:
                ids.append(index)
                scores.append(np.nan if score == "NA" else score)
                explanations.append(explanation)
            else:
                msg = f"Warning: Invalid score '{score}' on line {i}: '{line}'"
                print(msg)
                log(msg)
                error_count += 1
        else:
            msg = f"\n*********************\n" \
                  f"Warning: Invalid format on line {i}: '{line}'.\n" \
                  f"The full response was: \n{assistant_reply}\n" \
                  f"*********************\n"
            print(msg)
            log(msg)
            error_count += 1
            break

        if index - 1 >= len(batch_tweets):
            msg = f"Found {index} tweets, but there are more lines. Assuming remaining lines are junk and ignoring them."
            log(msg)
            print(msg)
            break

    if error_count > 0:
        scores = [np.nan] * len(batch_tweets)
        ids = [x + 1 for x in range(len(batch_tweets))]
        explanations = ["ERROR"] * len(batch_tweets)

    # Compute DataFrame indices
    df_ids = [batch_index + int(id) - 1 for id in ids]

    assert len(df_ids) == len(scores), f"Length of ids ({len(df_ids)}) does not match length of scores ({len(scores)})."

    return df_ids, scores, explanations

    
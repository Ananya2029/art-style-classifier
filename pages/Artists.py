import streamlit as st
import pandas as pd
import pickle
import requests
import time
from urllib.parse import quote


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Art Style Classifier",
    page_icon="🎨",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

artists = pd.read_csv("data/artists.csv")

with open("artifacts/similarity.pkl", "rb") as f:
    cosine_sim = pickle.load(f)


# =========================================================
# ARTIST RECOMMENDATION
# =========================================================

def recommend(artist_name):

    matches = artists.index[
        artists["name"] == artist_name
    ].tolist()

    if not matches:
        return []

    artist_idx = matches[0]

    sim_scores = list(
        enumerate(cosine_sim[artist_idx])
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_indices = [
        i[0]
        for i in sim_scores[1:5]
    ]

    return artists["name"].iloc[
        sim_indices
    ].tolist()


# =========================================================
# WIKIPEDIA API
#
# Searches Wikipedia instead of requiring an exact
# artist-name match.
#
# Example:
# Vasiliy Kandinskiy
#       ↓
# Wikipedia Search
#       ↓
# Wassily Kandinsky
# =========================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def get_artist_info(artist_name):

    api_url = (
        "https://en.wikipedia.org/w/api.php"
    )

    headers = {
        "User-Agent": (
            "ArtStyleClassifier/1.0 "
            "(educational project)"
        )
    }

    # =====================================================
    # STEP 0
    # Try the exact Wikipedia page we already know about,
    # from data/artists.csv's "wikipedia" column, before
    # falling back to a fuzzy live search. This matters
    # because several artists in the dataset are stored
    # under a display name that differs from their actual
    # Wikipedia title (e.g. "Rene Magritte" -> the real
    # page is "René Magritte"; "Salvador Dali" -> "Salvador
    # Dalí"), which a plain-text search can miss or mismatch.
    # =====================================================

    known_title = None

    try:

        match = artists[
            artists["name"] == artist_name
        ]

        if not match.empty:

            known_url = match.iloc[0].get(
                "wikipedia"
            )

            if (
                isinstance(known_url, str)
                and "/wiki/" in known_url
            ):

                known_title = known_url.split(
                    "/wiki/"
                )[-1]

    except Exception:

        known_title = None

    if known_title:

        for attempt in range(2):

            try:

                summary_url = (
                    "https://en.wikipedia.org/"
                    "api/rest_v1/page/summary/"
                    + quote(known_title, safe="")
                )

                summary_response = requests.get(
                    summary_url,
                    headers=headers,
                    timeout=10
                )

                if summary_response.status_code == 429:

                    # Brief backoff, then retry once before
                    # giving up on the direct lookup. Do NOT
                    # return early here - a transient rate
                    # limit on this one request should not
                    # make the whole lookup fail when we
                    # already know the correct page.
                    if attempt == 0:

                        time.sleep(1.5)
                        continue

                    break

                if summary_response.status_code == 200:

                    summary = summary_response.json()

                    # A disambiguation page has no useful
                    # extract/image for this purpose - fall
                    # through to the search-based path below.
                    if summary.get("type") != "disambiguation":

                        extract = summary.get(
                            "extract",
                            ""
                        )

                        image_url = None

                        if summary.get("originalimage"):

                            image_url = summary[
                                "originalimage"
                            ].get("source")

                        if (
                            not image_url
                            and summary.get("thumbnail")
                        ):

                            image_url = summary[
                                "thumbnail"
                            ].get("source")

                        full_url = (
                            summary
                            .get("content_urls", {})
                            .get("desktop", {})
                            .get("page")
                        )

                        if not full_url:

                            full_url = (
                                "https://en.wikipedia.org/wiki/"
                                + quote(known_title, safe="")
                            )

                        return {
                            "status": "success",
                            "title": summary.get(
                                "title",
                                artist_name
                            ),
                            "extract": extract,
                            "image": image_url,
                            "full_url": full_url
                        }

                # Any other status (404, 5xx, etc.), or a
                # disambiguation page - the known URL didn't
                # pan out this time. Stop retrying and fall
                # through to the search fallback below.
                break

            except requests.RequestException:

                break

            except ValueError:

                break

    try:

        # =================================================
        # STEP 1
        # Search Wikipedia
        # =================================================

        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": artist_name,
            "format": "json",
            "srlimit": 5
        }

        search_response = requests.get(
            api_url,
            params=search_params,
            headers=headers,
            timeout=10
        )

        # Wikipedia rate limit
        if search_response.status_code == 429:

            return {
                "status": "rate_limited"
            }

        search_response.raise_for_status()

        search_data = (
            search_response.json()
        )

        search_results = (
            search_data
            .get("query", {})
            .get("search", [])
        )

        # No search results
        if not search_results:

            return None

        # =================================================
        # STEP 2
        # Get best Wikipedia result
        # =================================================

        wikipedia_title = (
            search_results[0]
            .get("title")
        )

        if not wikipedia_title:

            return None

        # =================================================
        # STEP 3
        # Wikipedia REST API
        #
        # This usually gives a better image and summary.
        # =================================================

        encoded_title = quote(
            wikipedia_title.replace(
                " ",
                "_"
            ),
            safe=""
        )

        summary_url = (
            "https://en.wikipedia.org/"
            "api/rest_v1/page/summary/"
            + encoded_title
        )

        summary_response = requests.get(
            summary_url,
            headers=headers,
            timeout=10
        )

        # Rate limit
        if summary_response.status_code == 429:

            return {
                "status": "rate_limited"
            }

        # =================================================
        # STEP 4
        # REST API SUCCESS
        # =================================================

        if summary_response.status_code == 200:

            summary = (
                summary_response.json()
            )

            # ---------------------------------------------
            # Biography
            # ---------------------------------------------

            extract = summary.get(
                "extract",
                ""
            )

            # ---------------------------------------------
            # Image
            # ---------------------------------------------

            image_url = None

            # Prefer original image
            if summary.get(
                "originalimage"
            ):

                image_url = (
                    summary[
                        "originalimage"
                    ].get("source")
                )

            # Otherwise thumbnail
            if (
                not image_url
                and summary.get(
                    "thumbnail"
                )
            ):

                image_url = (
                    summary[
                        "thumbnail"
                    ].get("source")
                )

            # ---------------------------------------------
            # Wikipedia URL
            # ---------------------------------------------

            full_url = (
                summary
                .get("content_urls", {})
                .get("desktop", {})
                .get("page")
            )

            if not full_url:

                full_url = (
                    "https://en.wikipedia.org/wiki/"
                    + quote(
                        wikipedia_title.replace(
                            " ",
                            "_"
                        )
                    )
                )

            # ---------------------------------------------
            # Return
            # ---------------------------------------------

            return {
                "status": "success",

                "title": summary.get(
                    "title",
                    wikipedia_title
                ),

                "extract": extract,

                "image": image_url,

                "full_url": full_url
            }

        # =================================================
        # STEP 5
        # REST API FAILED
        #
        # Use normal Wikipedia API as fallback.
        # =================================================

        params = {

            "action": "query",

            "titles": wikipedia_title,

            "prop": (
                "extracts|pageimages|info"
            ),

            "inprop": "url",

            "format": "json",

            "exintro": True,

            "explaintext": True,

            "redirects": 1,

            "pithumbsize": 1000
        }

        response = requests.get(
            api_url,
            params=params,
            headers=headers,
            timeout=10
        )

        if response.status_code == 429:

            return {
                "status": "rate_limited"
            }

        response.raise_for_status()

        data = response.json()

        pages = (
            data
            .get("query", {})
            .get("pages", {})
        )

        page = next(
            iter(pages.values()),
            None
        )

        if (
            not page
            or "missing" in page
        ):

            return None

        # =================================================
        # IMAGE
        # =================================================

        image_url = (
            page
            .get("thumbnail", {})
            .get("source")
        )

        # =================================================
        # BIOGRAPHY
        # =================================================

        extract = page.get(
            "extract",
            ""
        )

        # =================================================
        # FULL WIKIPEDIA URL
        # =================================================

        full_url = page.get(
            "fullurl"
        )

        if not full_url:

            full_url = (
                "https://en.wikipedia.org/wiki/"
                + quote(
                    wikipedia_title.replace(
                        " ",
                        "_"
                    )
                )
            )

        # =================================================
        # RETURN
        # =================================================

        return {

            "status": "success",

            "title": page.get(
                "title",
                wikipedia_title
            ),

            "extract": extract,

            "image": image_url,

            "full_url": full_url
        }

    except requests.RequestException:

        return {
            "status": "error"
        }

    except ValueError:

        return {
            "status": "error"
        }


# =========================================================
# SHOW RECOMMENDED ARTISTS
# =========================================================

def display_recommendations(
    artist_name
):

    recommendations = recommend(
        artist_name
    )

    if not recommendations:

        return

    st.markdown(
        "## Recommended Artists"
    )

    cols = st.columns(
        len(recommendations)
    )

    for idx, artist in enumerate(
        recommendations
    ):

        with cols[idx]:

            # =================================================
            # GET WIKIPEDIA INFORMATION
            #
            # A small stagger between lookups avoids firing
            # several Wikipedia requests in a tight burst,
            # which can trigger short-lived rate limiting.
            # =================================================

            if idx > 0:

                time.sleep(0.4)

            artist_info = (
                get_artist_info(
                    artist
                )
            )

            # =================================================
            # ARTIST IMAGE
            # =================================================

            if (
                artist_info
                and artist_info.get(
                    "status"
                ) == "success"
                and artist_info.get(
                    "image"
                )
            ):

                st.image(
                    artist_info["image"],
                    width="stretch"
                )

            else:

                st.info(
                    "No image available"
                )

            # =================================================
            # ARTIST NAME
            # =================================================

            st.markdown(
                f"**{artist}**"
            )

            # =================================================
            # WIKIPEDIA BUTTON
            # =================================================

            if (
                artist_info
                and artist_info.get(
                    "full_url"
                )
            ):

                wikipedia_url = (
                    artist_info[
                        "full_url"
                    ]
                )

            else:

                wikipedia_url = (
                    "https://en.wikipedia.org/wiki/"
                    + quote(
                        artist.replace(
                            " ",
                            "_"
                        )
                    )
                )

            st.link_button(
                "Wikipedia",
                wikipedia_url,
                use_container_width=True
            )

            # =================================================
            # MORE ABOUT BUTTON
            # =================================================

            if st.button(
                f"More about {artist}",
                key=(
                    f"more_about_"
                    f"{idx}_"
                    f"{artist}"
                ),
                use_container_width=True
            ):

                # Store selected artist
                st.session_state[
                    "selected_artist"
                ] = artist

                # Rerun application
                st.rerun()


# =========================================================
# SHOW SELECTED ARTIST
# =========================================================

def show_artist_info(
    artist_name
):

    artist_info = (
        get_artist_info(
            artist_name
        )
    )

    # =====================================================
    # WIKIPEDIA RATE LIMITED
    # =====================================================

    if (
        artist_info
        and artist_info.get(
            "status"
        ) == "rate_limited"
    ):

        st.title(
            artist_name
        )

        st.warning(
            "Wikipedia is temporarily "
            "rate-limiting requests."
        )

        wikipedia_url = (
            "https://en.wikipedia.org/wiki/"
            + quote(
                artist_name.replace(
                    " ",
                    "_"
                )
            )
        )

        st.link_button(
            "Read about this artist on Wikipedia",
            wikipedia_url
        )

    # =====================================================
    # WIKIPEDIA CONNECTION ERROR
    # =====================================================

    elif (
        artist_info
        and artist_info.get(
            "status"
        ) == "error"
    ):

        st.title(
            artist_name
        )

        st.warning(
            "Artist information is temporarily "
            "unavailable."
        )

        wikipedia_url = (
            "https://en.wikipedia.org/wiki/"
            + quote(
                artist_name.replace(
                    " ",
                    "_"
                )
            )
        )

        st.link_button(
            "Read about this artist on Wikipedia",
            wikipedia_url
        )

    # =====================================================
    # SUCCESS
    # =====================================================

    elif artist_info:

        # =================================================
        # TITLE
        # =================================================

        st.title(
            artist_info[
                "title"
            ]
        )

        # =================================================
        # TWO COLUMNS
        # =================================================

        col1, col2 = st.columns(
            [1, 2]
        )

        # =================================================
        # ARTIST IMAGE
        # =================================================

        with col1:

            if (
                artist_info.get(
                    "image"
                )
            ):

                st.image(
                    artist_info[
                        "image"
                    ],
                    width="stretch"
                )

            else:

                st.info(
                    "No artist image available."
                )

        # =================================================
        # BIOGRAPHY
        # =================================================

        with col2:

            extract = (
                artist_info.get(
                    "extract",
                    ""
                )
            )

            words = extract.split()

            biography = " ".join(
                words[:120]
            )

            if len(words) > 120:

                biography += "..."

            st.subheader(
                "Biography"
            )

            if biography:

                st.write(
                    biography
                )

            else:

                st.write(
                    "Biography information "
                    "is not available."
                )

            # =================================================
            # READ MORE BUTTON
            # =================================================

            st.link_button(
                "Read more on Wikipedia",
                artist_info[
                    "full_url"
                ]
            )

    # =====================================================
    # NO INFORMATION
    # =====================================================

    else:

        st.title(
            artist_name
        )

        st.warning(
            "No information found for this artist."
        )

        wikipedia_url = (
            "https://en.wikipedia.org/wiki/"
            + quote(
                artist_name.replace(
                    " ",
                    "_"
                )
            )
        )

        st.link_button(
            "Read about this artist on Wikipedia",
            wikipedia_url
        )

    # =====================================================
    # RECOMMENDED ARTISTS
    # =====================================================

    display_recommendations(
        artist_name
    )


# =========================================================
# SIDEBAR
# =========================================================

artist_list = (
    ["Select an artist"]
    + sorted(
        artists[
            "name"
        ].unique()
    )
)


# =========================================================
# GET CURRENT ARTIST
# =========================================================

current_artist = (
    st.session_state.get(
        "selected_artist",
        "Select an artist"
    )
)


# =========================================================
# VALIDATE CURRENT ARTIST
# =========================================================

if current_artist not in artist_list:

    current_artist = (
        "Select an artist"
    )


# =========================================================
# SIDEBAR SELECTBOX
# =========================================================

selected_artist = (
    st.sidebar.selectbox(
        "Select an artist",
        artist_list,
        index=artist_list.index(
            current_artist
        )
    )
)


# =========================================================
# DETECT MANUAL SIDEBAR SELECTION
# =========================================================

if (
    selected_artist
    != "Select an artist"
):

    st.session_state[
        "selected_artist"
    ] = selected_artist

elif (
    "selected_artist"
    not in st.session_state
):

    st.session_state[
        "selected_artist"
    ] = "Select an artist"


# =========================================================
# MAIN PAGE
# =========================================================

if (
    st.session_state[
        "selected_artist"
    ]
    != "Select an artist"
):

    show_artist_info(
        st.session_state[
            "selected_artist"
        ]
    )

else:

    st.title(
        "🎨 WHO IS YOUR FAVORITE ARTIST?"
    )

    st.subheader(
        "Select an artist from Renaissance "
        "to Modern and discover similar artists."
    )

    import os
    if os.path.exists("assets/gogh.png"):
        st.image(
            "assets/gogh.png"
        )
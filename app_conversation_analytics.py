import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import utils


st.title("Analyse automatique des conversations Butler")


# =========================
# Utils app
# =========================

def read_file(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)

    elif file.name.endswith(".csv"):
        for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
            try:
                file.seek(0)
                return pd.read_csv(file, encoding=encoding)
            except Exception:
                pass

        raise ValueError(f"Impossible de lire {file.name}")


def parse_started_date(df):
    return pd.to_datetime(
        df["started"].str.replace(" Europe/Paris", "", regex=False),
        errors="coerce"
    ).dt.tz_localize("Europe/Paris")


def build_readable_transcript(t):
    tmp = t.copy()

    tmp["speaker_type"] = tmp["speaker"].apply(utils.classify_speaker)

    speaker_label = {
        "user": "USER",
        "bot": "BOT",
        "human_agent": "AGENT",
        "csat": "CSAT",
        "other": "OTHER",
        "exclude": "EXCLUDE"
    }

    tmp["speaker_label"] = tmp["speaker_type"].map(speaker_label).fillna("OTHER")

    tmp["line"] = (
        "["
        + tmp["heure_message_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
        + "] "
        + tmp["speaker_label"]
        + " : "
        + tmp["message"].fillna("").astype(str)
    )

    return (
        tmp.groupby("id")["line"]
        .apply(lambda x: "\n".join(x))
        .reset_index()
        .rename(columns={"line": "transcript_lisible"})
    )


def agent_positive_feedback_analysis(t):
    """
    Détecte si l'utilisateur donne un feedback positif après une intervention agent.
    """
    x = t.copy()
    x["speaker_type"] = x["speaker"].apply(utils.classify_speaker)
    x = x.sort_values(["id", "question_id", "ordre_message_question"])

    positive_pattern = re.compile(
        r"""
        (
            \boui\b
            | merci\b
            | merci\s+beaucoup
            | parfait\b
            | super\b
            | top\b
            | nickel\b
            | impeccable\b
            | ça\s+marche
            | ca\s+marche
            | c['’]est\s+bon
            | c['’]est\s+ok
            | résolu\b
            | \byes\b
            | thanks?\b
            | thank\s+you
            | perfect\b
            | great\b
            | awesome\b
            | amazing\b
            | all\s+good
            | resolved\b
            | problem\s+solved
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )

    x["human_agent_has_spoken_before"] = (
        (x["speaker_type"] == "human_agent")
        .groupby([x["id"], x["question_id"]])
        .cummax()
        .shift(1)
        .fillna(False)
    )

    x["user_positive_feedback_after_agent"] = (
        (x["speaker_type"] == "user")
        & x["human_agent_has_spoken_before"]
        & x["message"].fillna("").str.contains(positive_pattern, regex=True)
    )

    return (
        x.groupby(["id", "question_id"])
        .agg(
            agent_positive_feedback=("user_positive_feedback_after_agent", "max")
        )
        .reset_index()
    )


def get_question_content(questions):
    q = questions.copy()
    q["speaker_type"] = q["speaker"].apply(utils.classify_speaker)

    first_user_question = (
        q[
            (q["speaker_type"] == "user")
            & q["question_id"].notna()
            & (q["is_new_question_start"])
        ]
        .sort_values(["id", "question_id", "ordre_message_question"])
        .groupby(["id", "question_id"])
        .agg(
            datetime_question=("heure_message_dt", "first"),
            question=("message", "first")
        )
        .reset_index()
    )

    fallback = (
        q[
            (q["speaker_type"] == "user")
            & q["question_id"].notna()
        ]
        .sort_values(["id", "question_id", "ordre_message_question"])
        .groupby(["id", "question_id"])
        .agg(
            datetime_question_fallback=("heure_message_dt", "first"),
            question_fallback=("message", "first")
        )
        .reset_index()
    )

    result = first_user_question.merge(
        fallback,
        how="outer",
        on=["id", "question_id"]
    )

    result["datetime_question"] = result["datetime_question"].fillna(
        result["datetime_question_fallback"]
    )

    result["question"] = result["question"].fillna(
        result["question_fallback"]
    )

    return result[["id", "question_id", "datetime_question", "question"]]


def build_analysis(conversations, start_date, end_date):
    conversations = conversations.copy()

    conversations["started_dt"] = parse_started_date(conversations)
    conversations["date"] = conversations["started_dt"].dt.date

    conversations = conversations[
        (conversations["date"] >= start_date)
        & (conversations["date"] <= end_date)
    ].copy()

    # 1. Transcript
    transcript = utils.get_transcript(conversations)

    # 2. Questions
    questions = utils.get_questions(transcript)

    # 3. Métriques question
    q_metrics = utils.question_gen_metrics(questions)

    # 4. Contenu de la question
    q_content = get_question_content(questions)

    # 5. Réponse bot
    bot_answer = utils.answer_analysis(questions)

    # 6. Escalade
    escalation = utils.escalation_analysis(questions)

    # 7. Réponse agent
    agent_reply = utils.agent_reply_analysis(questions)

    # 8. Feedback positif après agent
    agent_feedback = agent_positive_feedback_analysis(questions)

    # 9. CSAT
    csat = utils.survey_evaluation(conversations)

    # 10. Transcript lisible
    readable_transcript = build_readable_transcript(questions)

    # 11. Merge final
    result = q_metrics.merge(q_content, how="left", on=["id", "question_id"])
    result = result.merge(bot_answer, how="left", on=["id", "question_id"])
    result = result.merge(escalation, how="left", on=["id", "question_id"])
    result = result.merge(agent_reply, how="left", on=["id", "question_id"])
    result = result.merge(agent_feedback, how="left", on=["id", "question_id"])
    result = result.merge(csat, how="left", on="id")
    result = result.merge(readable_transcript, how="left", on="id")

    # 12. Nettoyage / renommage
    result["date_question"] = pd.to_datetime(result["datetime_question"]).dt.date
    result["heure_question"] = pd.to_datetime(result["datetime_question"]).dt.time

    result["agent_a_pu_repondre"] = result["question_status"].isin([
        "ticket_created",
        "redirected_to_form",
        "resolved_by_agent"
    ])

    result["comment_agent_a_repondu"] = result["question_status"].map({
        "ticket_created": "Ticket créé",
        "redirected_to_form": "Redirection vers formulaire",
        "resolved_by_agent": "Résolution par agent",
        "not_resolved": "Agent non en mesure d'aider",
        "unknown": "Inconnu"
    })

    result["csat_answered"] = result["has_survey"].fillna(False)

    result["csat_feedback"] = result["survey_feedback"].replace({
        "thumbs_up": "positive",
        "thumbs_down": "negative",
        "none": "none",
        False: "none"
    })

    final = result.rename(columns={
        "id": "id_conv",
        "nb_msg_question": "nb_msg_question",
        "bot_answer": "bot_a_repondu",
        "bot_answer_satisfying": "feedback_positif_utilisateur_apres_reponse_bot",
        "escalation_offered": "escalade_proposee",
        "escalation_effective": "escalade_effective",
        "agent_positive_feedback": "feedback_positif_utilisateur_apres_reponse_agent",
    })

    final = final[
        [
            "id_conv",
            "question_id",
            "hotelCode",
            "nb_msg_question",
            "date_question",
            "heure_question",
            "question",
            "bot_a_repondu",
            "feedback_positif_utilisateur_apres_reponse_bot",
            "escalade_proposee",
            "escalade_effective",
            "agent_a_pu_repondre",
            "comment_agent_a_repondu",
            "feedback_positif_utilisateur_apres_reponse_agent",
            "csat_answered",
            "csat_feedback",
            "transcript_lisible"
        ]
    ]

    bool_cols = [
        "bot_a_repondu",
        "feedback_positif_utilisateur_apres_reponse_bot",
        "escalade_proposee",
        "escalade_effective",
        "agent_a_pu_repondre",
        "feedback_positif_utilisateur_apres_reponse_agent",
        "csat_answered",
    ]

    for col in bool_cols:
        final[col] = final[col].fillna(False)

    return final


# =========================
# App Streamlit
# =========================

export_conv = st.file_uploader(
    "Export Conversations",
    type=["xlsx", "csv"]
)

if export_conv:
    conversations = read_file(export_conv)

    required_cols = [
        "id",
        "started",
        "transcript",
        "hotelCode",
        "customerHandle",
        "assignee"
    ]

    missing_cols = [col for col in required_cols if col not in conversations.columns]

    if missing_cols:
        st.error(f"Colonnes manquantes dans le fichier : {missing_cols}")
        st.stop()

    conv_dates = parse_started_date(conversations)

    min_data_date = conv_dates.min().date()
    max_data_date = conv_dates.max().date()

    start_date = st.date_input(
        "Date de début d'analyse",
        value=min_data_date,
        min_value=min_data_date,
        max_value=max_data_date
    )

    end_date = st.date_input(
        "Date de fin d'analyse",
        value=max_data_date,
        min_value=min_data_date,
        max_value=max_data_date
    )

    if start_date > end_date:
        st.error("La date de début ne peut pas être après la date de fin.")
        st.stop()

    st.success("Fichier chargé")

    st.write("Aperçu des conversations")
    st.dataframe(conversations.head())

    if st.button("Lancer l'analyse"):

        result = build_analysis(
            conversations,
            start_date,
            end_date
        )

        st.success("Analyse terminée")

        st.write("Résultat")
        st.dataframe(result)

        output = io.BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result.to_excel(writer, index=False, sheet_name="questions_analysis")

        output.seek(0)

        st.download_button(
            label="Télécharger les résultats",
            data=output,
            file_name="questions_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

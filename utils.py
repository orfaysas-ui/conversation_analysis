import pandas as pd
import re
import numpy as np

def clean_dates(d):
    r = d[['id','started']]
    r['date']=pd.to_datetime(r["started"].str.replace(" Europe/Paris", "", regex=False)).dt.tz_localize("Europe/Paris").dt.date
    r['year']=pd.to_datetime(r.date).dt.year
    r['month']=pd.to_datetime(r.date).dt.month
    return r
    
def get_hotel_code(d):
    r = d.groupby('Conversation ID')['Hotel Code'].first().reset_index().rename(columns={'Hotel Code':'hotel_code','Conversation ID':'id'})
    return r

def get_topic(d):
    r = d.groupby('Conversation ID')['Topic'].first().reset_index().rename(columns={'Topic':'topic','Conversation ID':'id'})
    return r

def is_blank(d):
    r = d[d.Question=='BLANK']
    return r


def get_thumb(x):
    text = " ".join(x.astype(str))
    if "👍" in text:
        return "thumbs_up"
    elif "👎" in text:
        return "thumbs_down"
    else:
        return "none"
    
def survey_evaluation(d):
    d['csat_asked']=d.transcript.str.contains("👍")
    d["is_survey"] = d["assignee"].eq("csat-survey")
    d["date_only"] = pd.to_datetime(
        d["started"].str.replace(" Europe/Paris", "", regex=False)
    ).dt.tz_localize("Europe/Paris").dt.date

    
    survey_flags = (
    d[d["is_survey"]].groupby(["customerHandle",'date_only'])
    .agg(
        has_survey=("id", "count"),
        survey_feedback=("transcript", get_thumb)
    )
    .reset_index()
    )
    
    survey_flags["has_survey"] = survey_flags["has_survey"].gt(0)
    
    d2 = (
    d[~d["is_survey"]]  # garde seulement les vraies conversations
    .merge(survey_flags, on=["customerHandle",'date_only'], how="left")
    )
    
    d2["has_survey"] = d2["has_survey"].fillna(False)
    d2["survey_feedback"] = d2["survey_feedback"].fillna(False)
    r = d2[['id','has_survey','survey_feedback']]
    
    return r

def classify_speaker(s):
    s = str(s).lower()
    if "auto-response" in s:
        return "exclude"
    if "agent (ai-butler)" in s:
        return "bot"
    if "consumer" in s:
        return "user"
    if ("agent (" in s) & ("butler" not in s):
        return "human_agent"
    if "csat" in s:
        return "csat"
    return "other"

def get_transcript(d):
    pattern = r"\[(.*?)\]\s*(.*?)\s*:\s*(.*?)(?=\n\[\d{2}/\d{2}/\d{4}|\Z)"
    
    pattern_msg = (
    r"(?s)\[(.*?)\]\s*([^:\n]+):\s*(.*?)"
    r"(?=\r?\n\[\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s+(?:AM|PM)\s+\w+\]|\Z)"
    )
    
    transcript = (
    d
    .assign(
        messages=d["transcript"].str.findall(pattern_msg)
    )
    .explode("messages")
    .dropna(subset=["messages"])
    )
    
    transcript[["heure_message", "speaker", "message"]] = pd.DataFrame(
    transcript["messages"].tolist(),
    index=transcript.index
    )
    
    transcript = transcript[(transcript.message !='Incoming Chat')
                &(transcript.speaker!='Auto-response')].drop(columns=["messages"]) 
    
    transcript["heure_message_dt"] = pd.to_datetime(
    transcript["heure_message"]
        .str.replace(" CEST", "", regex=False)
        .str.replace(" CET", "", regex=False),
    format="%m/%d/%Y %I:%M:%S %p"
    )
    
    transcript = transcript.sort_values(["id", "heure_message_dt"])
    transcript["ordre_message_conv"] = transcript.groupby("id").cumcount() + 1
        
    return transcript




def conv_gen_metrics(t):
    r=t.groupby(['id','customerHandle']).agg(
    nb_questions = ('question_id','nunique'),
    nb_msg_conv = ('ordre_message_conv','max')).reset_index()
    
    return r

def bot_answer_analysis(q) : 
    
    r=q.copy()
    
    #bot asking feedback
    
    pattern_feedback = re.compile(
    r"""
    (?:
        # Français
        cela\s+vous\s+a[-\s]?t[-\s]?il\s+aid
        |
        est[-\s]?ce\s+que\s+cela\s+vous\s+a\s+aid
        |
        # Anglais
        did\s+this\s+help
        |
        did\s+that\s+help
        |
        was\s+this\s+helpful
        |
        did\s+this\s+resolve
        |
        did\s+this\s+solve
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
    )
    
    r["bot_asks_feedback"] = (
        (r.speaker.apply(classify_speaker) == 'bot')
        & r["message"].fillna("").str.contains(pattern_feedback, regex=True)
    )
    
    #bot has asked feedback 
    
    # tracking message user
    
    r["user_msg_counter"] = (
        (r["speaker_type"] == "user")
        .groupby(r["id"])
        .cumsum()
    )
    

    r["between_user_block"] = r["user_msg_counter"]

    r["bot_asked_fb_since_last_user"] = (
        r.groupby(["id", "between_user_block"])["bot_asks_feedback"]
        .cummax()
        .shift(fill_value=False)
    )
    
    r["bot_asked_fb_since_last_user"] = (
    (r["bot_asked_fb_since_last_user"])&(r.speaker_type=='user')
    )
    
    #user feedback is positive
    
    pattern_user_feedback_positive = re.compile(
    r"""
    (
        # FR
        \boui\b
        | merci\b
        | merci\s+beaucoup
        | super\b
        | génial\b
        | parfait\b
        | top\b
        | nickel\b
        | c['’]est\s+bon
        | ça\s+marche
        | ca\s+marche
        | c['’]est\s+ok
        | tout\s+est\s+bon
        | c['’]est\s+parfait
        | impeccable
        | résolu
        | problème\s+résolu
        
        |
        
        # EN
        \byes\b
        | thanks?\b
        | thank\s+you
        | awesome\b
        | great\b
        | perfect\b
        | amazing\b
        | cool\b
        | nice\b
        | it\s+works
        | works?\s+fine
        | all\s+good
        | resolved\b
        | problem\s+solved
        | that\s+helped
        | this\s+helped
        | ok\b
        | okay\b
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )
    
    r['user_positive_feedback']=(
        (r.speaker_type=='user')
        &(r["message"].fillna("").str.contains(pattern_user_feedback_positive, regex=True))
        &r['bot_asked_fb_since_last_user']
    )
    
    return(r)
    
def question_gen_metrics(t):
    r=(t.groupby(['id','question_id','hotelCode']).agg(
        nb_msg_question = ('ordre_message_question','max')
                 )
        .reset_index())
    
    return r

def escalation_analysis(t):
    
    pattern_escalade = re.compile(
    r"""
    (?:
        # FR / EN : mots-clés de relais humain
        agent\s+humain
        |
        human\s+(?:agent|expert)
        |
        expert\s+humain
        |
        human\s+support
        |
        expert\s+heartist
        |
        heartist
        |
        agents?\s+experts?
        |
        experts?\s+heartist
        |
        conseiller\s+humain
        |
        membre\s+de\s+(?:notre\s+)?équipe
        |
        one\s+of\s+our\s+(?:human\s+)?experts?
        |
        human\s+expert
        |
        take\s+a\s+look\s+at\s+this
        |
        prendre\s+le\s+relais
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
    )
    
    t['offers_escalation']=(t.speaker.apply(classify_speaker) =='bot')&(t.message.str.contains(pattern_escalade, na=False))&(t.ordre_message_conv>2)
    
    pattern_acceptation = re.compile(
    r"""
    (
        \b(oui|yes|yep|yeah|ok|okay)\b
        |
        \b(d['’]?accord|volontiers|bien sûr|bien sur)\b
        |
        \b(je\s+veux\s+bien|allez-y|vas-y|go\s+ahead|please|yes\s+please)\b
        |
        \b(connectez[-\s]?moi|mettez[-\s]?moi\s+en\s+relation|transférez[-\s]?moi)\b
        |
        \b(connect\s+me|transfer\s+me|put\s+me\s+through)\b
        |
        parler\s+à\s+(un\s+)?(expert|agent|humain)
        |
        talk\s+to\s+(an?\s+)?(agent|human|expert)
        |
        \b(agent|expert|humain|human)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )
    
    t = t.sort_values(["id", "ordre_message_conv"])
    
    t["prev_escalade"] = (
    t.groupby("id")["offers_escalation"]
    .shift(1)
    .fillna(False)
    )
    
    t["acceptation_escalade"] = (
        t["prev_escalade"]
        &
        (t.speaker.apply(classify_speaker)=='user')
        &
        (t["message"].str.contains(pattern_acceptation, na=False))
    )
    
    t['human_agent']=(t.speaker.apply(classify_speaker)=='human_agent')
    
    
    r = (
    t.groupby(["id", "question_id"])
    .agg(
        escalation_offered=('offers_escalation','max'),
        escalation_accepted=('acceptation_escalade','max'),
        escalation_effective=('human_agent','max'),
    )
    .reset_index()
    )
    
    return r

def answer_analysis(x):
    
    t=x.copy()
    
    pattern_feedback = re.compile(
    r"""
    (?:
        # Français
        cela\s+vous\s+a[-\s]?t[-\s]?il\s+aid
        |
        est[-\s]?ce\s+que\s+cela\s+vous\s+a\s+aid
        |
        # Anglais
        did\s+this\s+help
        |
        did\s+that\s+help
        |
        was\s+this\s+helpful
        |
        did\s+this\s+resolve
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
    )
    
    pattern_no_answer = re.compile(
    r"""
    (?:
        # Français
        je\s+n['’]ai\s+pas\s+trouv[ée]?\s+d['’]informations?
        |
        je\s+n['’]ai\s+pas\s+trouv[ée]?\s+d['’]informations?\s+spécifiques?
        |
        je\s+n['’]ai\s+pas\s+trouv[ée]?\s+d['’]information\s+sur
        |
        aucune\s+information\s+spécifique
        |
        information\s+non\s+trouv[ée]e?

        |

        # Anglais
        i\s+could\s+not\s+find\s+information
        |
        i\s+couldn['’]t\s+find\s+information
        |
        i\s+did\s+not\s+find\s+information
        |
        i\s+don['’]t\s+have\s+information
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
    )
    
    t["bot_feedback_request"] = (
    (t.speaker.apply(classify_speaker) == "bot")
    & t["message"].str.contains(pattern_feedback, na=False)
    )

    t["bot_no_answer"] = (
    (t.speaker.apply(classify_speaker) == "bot")
    & t["message"].str.contains(pattern_no_answer, na=False)
    )
    
    t["prev_bot_no_answer"] = (
    t.groupby("id")["bot_no_answer"]
    .shift(1)
    .fillna(False)
    )
    
    t["bot_feedback_after_answer"] = (
        t["bot_feedback_request"]
        & ~t["prev_bot_no_answer"]
    )
    
    
    pattern_feedback_positif = re.compile(
    r"""
    (?:
        \b(?:oui|yes|yeah|yep)\b
        |
        (?:c'?est|c\s+est)\s+(?:parfait|bon|clair|nickel|top|super)
        |
        (?:ça|ca|cela)\s+(?:m'?a\s+)?(?:aidé|aide|helped)
        |
        (?:that|this)\s+(?:helped|works)
        |
        (?:perfect|great|thanks|thank\s+you|merci)
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
    )
    
    pattern_feedback_negatif = re.compile(
    r"""
    (?:
        \b(?:non|no|nope)\b
        |
        (?:ça|ca|cela)\s+(?:ne\s+)?(?:m'?a\s+)?pas\s+(?:aidé|aide)
        |
        (?:not|doesn'?t|didn'?t)\s+(?:help|work|answer)
        |
        (?:ce\s+n'?est\s+pas|c'?est\s+pas)\s+(?:clair|bon|utile)
        |
        (?:i\s+still|je\s+ne\s+comprends\s+toujours|je\s+n'?ai\s+toujours)
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
    )
    
    t = t.sort_values(["id", "ordre_message_question"])
    
    t["prev_feedback_request"] = (
    t.groupby("id")["bot_feedback_request"]
    .shift(1)
    .fillna(False)
    )
    
    t["feedback_reponse_positive"] = (
        t["prev_feedback_request"]
        &
        (t.speaker.apply(classify_speaker)=='user')
        &
        (t["message"].str.contains(pattern_feedback_positif, na=False))
    )
    
    t["feedback_reponse_negative"] = (
        t["prev_feedback_request"]
        &
        (t.speaker.apply(classify_speaker)=='user')
        &
        (t["message"].str.contains(pattern_feedback_negatif, na=False))
    )
    
    r = (
    t.groupby(["id", "question_id"])
    .agg(
        bot_answer=("bot_feedback_after_answer", "max"),
        bot_answer_satisfying=("feedback_reponse_positive", "max"),
        bot_answer_unsatisfying=("feedback_reponse_negative", "max"),
    )
    .reset_index()
    )

    return r

def agent_reply_analysis (t) : 
    
    t["is_human_agent"] = (t.speaker.apply(classify_speaker)=='human_agent' )
    
    pattern_resolution = re.compile(
    r"""
    (
        est[- ]ce\s+que\s+(cela|ça)\s+vous\s+a\s+aid
        |
        est[- ]ce\s+que\s+j['’]ai\s+pu\s+vous\s+aider
        |
        ravi\s+d['’]avoir\s+pu\s+vous\s+aider
        |
        heureux\s+d['’]avoir\s+pu\s+vous\s+aider
        |
        y\s+a[- ]t[- ]il\s+autre\s+chose
        |
        puis[- ]je\s+faire\s+autre\s+chose
        |
        ai[- ]je\s+répondu\s+à\s+votre\s+question
        |
        does\s+this\s+help
        |
        did\s+this\s+help
        |
        was\s+i\s+able\s+to\s+help
        |
        anything\s+else\s+i\s+can\s+help
        |
        can\s+i\s+help\s+with\s+anything\s+else
        |
        glad\s+i\s+could\s+help
    )
    """,
    re.I | re.X
    )
    
    t["agent_resolution_attempt"] = (
    t["is_human_agent"]
    & t["message"].str.contains(pattern_resolution, na=False)
    )
    
    pattern_ticket = re.compile(r"\bCS\d{8}\b", re.I)
    
    t["agent_created_ticket"] = (
    t["is_human_agent"]
    & t["message"].str.contains(pattern_ticket, na=False)
    )
    
    t["ticket_id"] = (
    t["message"]
    .str.extract(r"\b(CS\d{8})\b", flags=re.I)[0]
    .str.upper()
    )
    
    pattern_form = re.compile(
    r"""
    (
        formulaire
        | form
        | survey
        | questionnaire
        | request
        | demande
        | ticket
        | veuillez\s+remplir
        | merci\s+de\s+compl[ée]ter
        | fill\s+(in|out)
        | submit
        | portal
        | portail
        | lien
        | link
    )
    """,
    re.I | re.X
    )
    
    pattern_url = r"https?://\S+|www\.\S+|\b\S+\.[a-zA-Z]{2,}\S*"
    
    t["agent_sent_form"] = (
    t["is_human_agent"]
    & t["message"].str.contains(pattern_url, na=False)
    & t["message"].str.contains(pattern_form, na=False)
    )
    
    pattern_failure = re.compile(
    r"""
    (
        pas\s+dans\s+mon\s+p[ée]rim[èe]tre
        |
        hors\s+de\s+mon\s+scope
        |
        je\s+ne\s+peux\s+pas
        |
        impossible\s+de
        |
        nous\s+ne\s+pouvons\s+pas
        |
        je\s+n['’]ai\s+pas\s+acc[èe]s
        |
        je\s+ne\s+suis\s+pas\s+en\s+mesure
        |
        ce\s+n['’]est\s+pas\s+possible
        |
        unfortunately
        |
        out\s+of\s+scope
        |
        i\s+cannot
        |
        i'm\s+unable\s+to
        |
        not\s+able\s+to
        |
        no\s+access\s+to
    )
    """,
    re.I | re.X
    )
    
    t["agent_could_not_help"] = (
    t["is_human_agent"]
    & t["message"].str.contains(pattern_failure, na=False)
    )
    
    agg = (
    t.groupby(["id", "question_id"])
    .agg(
        resolved=("agent_resolution_attempt", "max"),
        ticket=("agent_created_ticket", "max"),
        form=("agent_sent_form", "max"),
        failed=("agent_could_not_help", "max"),
    )
    .reset_index()
    )
    
    conditions = [
        agg["ticket"],
        agg["form"],
        agg["resolved"],
        agg["failed"],
    ]
    
    choices = [
        "ticket_created",
        "redirected_to_form",
        "resolved_by_agent",
        "not_resolved",
    ]

    agg["question_status"] = np.select(
        conditions,
        choices,
        default="unknown"
    )
    
    return agg
    
def get_questions(x):
    
    t = x[~x.assignee.str.contains('csat')]
    t = t.sort_values(["id", "ordre_message_conv"])

    t["speaker_type"] = t["speaker"].apply(classify_speaker)
    t["message_clean"] = t["message"].fillna("")
    
    #First user message (besides greetings)
    
    
    pattern_greeting_only = re.compile(
    r"""
    ^\s*
    (
        bonjour
        | hello
        | hi
        | hey
        | bonsoir
        | salut
        | good\s+morning
        | good\s+evening
        | oui
        | non
        | yes
        | no
    )
    [\s,!.?-]*
    
    (
        h\d+
    )?
    
    [\s,!.?-]*$
    """,
    re.IGNORECASE | re.VERBOSE
    )

    t["user_greeting_only"] = (
        (t["speaker_type"] == "user")
        & t["message"].fillna("").str.contains(pattern_greeting_only, regex=True)
    )
    
    first_user_msg = (
        t[(t["speaker_type"] == "user")&(~t["user_greeting_only"])]
        .groupby("id")["ordre_message_conv"]
        .min()
        .reset_index()
        .rename(columns={"ordre_message_conv": "first_user_message"})
    )
    
    t = t.merge(first_user_msg, how="left", on="id")
    
    t["has_user_message"] = t["first_user_message"].notna()
    
    #first bot message
    
    first_bot_msg = (
        t[t["speaker_type"] == "bot"]
        .groupby("id")["ordre_message_conv"]
        .min()
        .reset_index()
        .rename(columns={"ordre_message_conv": "first_bot_message"})
    )

    t = t.merge(first_bot_msg, how="left", on="id")
    
    
    #first human intervention
    
    t["is_human_agent"] = t["speaker"].apply(classify_speaker).eq("human_agent")

    t["human_agent_has_spoken_before"] = (
    t.groupby("id")["is_human_agent"]
    .cummax()
    .shift(1)
    .fillna(False)
    )
    

    # Requete utilisateur
    pattern_user_request = re.compile(
        r"""
        (
            \?
            |
            \b(
                donne[z]?\s+moi
                | dis[\s-]?moi
                | indique[z]?\s+moi
                | explique[z]?\s+moi
                | montre[z]?\s+moi
                | aide[z]?\s+moi
                | peux[\s-]?tu
                | pourrais[\s-]?tu
                | pouvez[\s-]?vous
                | pourriez[\s-]?vous
                | je\s+veux
                | je\s+voudrais
                | j'aimerais
                | je\s+souhaiterais
                | je\s+souhaite
                | je\s+cherche
                | j['’]ai\s+besoin
                | besoin\s+d['’]aide
                | je\s+n['’]arrive\s+pas
                | impossible\s+de
                | marche\s+pas
                | fonctionne\s+pas
                | arrive\s+pas
                | comment
                | pourquoi
                | ou
                | où
                | quand
                | combien
                | que\s+faire
                | quoi\s+faire
                | qu['’]est[\s-]?ce\s+que
                | est[\s-]?ce\s+que
                | un\s+problème
                | un\s+souci
                | un\s+pb
                | i\s+need
                | i\s+want
                | i\s+would\s+like
                | i\s+wish
                | can\s+you
                | could\s+you
                | tell\s+me
                | give\s+me
                | show\s+me
                | explain
                | help\s+me
                | how\s+do\s+i
                | how\s+can\s+i
                | what\s+should\s+i
                | why
                | where
                | when
                | issue
                | problem
                | stuck
            )\b
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )

    t["user_message_is_request"] = (
        (t["speaker_type"] == "user")
        & t["message_clean"].str.contains(pattern_user_request, na=False)
    )

    # Bot demande si il peut aider à autre chose
    pattern_anything_else = re.compile(
        r"""
        (
            autre\s+chose
            | autre\s+question
            | autre\s+sujet
            | autre\s+probl[eè]me
            | puis[-\s]?je\s+vous\s+aider
            | est[-\s]?ce\s+que\s+je\s+peux\s+vous\s+aider
            | je\s+peux\s+vous\s+aider
            | je\s+suis\s+l[àa]\s+si\s+besoin
            | n['’]h[eé]sitez\s+pas
            | besoin\s+d['’]aide
            | vous\s+aider\s+davantage
            | can\s+i\s+help
            | anything\s+else
            | something\s+else
            | do\s+you\s+need\s+help
            | i['’]?m\s+here\s+if\s+you\s+need
            | let\s+me\s+know\s+if
            | happy\s+to\s+help
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )

    t["bot_asks_anything_else"] = (
        (t["speaker_type"] == "bot")
        & t["message_clean"].str.contains(pattern_anything_else, na=False)
    )

    # Bot propose d'escalader
      
    pattern_escalade = re.compile(
    r"""
    (?:
        # FR / EN : mots-clés de relais humain
        agent\s+humain
        |
        human\s+(?:agent|expert)
        |
        expert\s+humain
        |
        human\s+support
        |
        expert\s+heartist
        |
        heartist
        |
        agents?\s+experts?
        |
        experts?\s+heartist
        |
        conseiller\s+humain
        |
        membre\s+de\s+(?:notre\s+)?équipe
        |
        one\s+of\s+our\s+(?:human\s+)?experts?
        |
        human\s+expert
        |
        take\s+a\s+look\s+at\s+this
        |
        prendre\s+le\s+relais
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
    )
    
    t['bot_proposed_escalation']=(
        (t.speaker.apply(classify_speaker) =='bot')
        &t.message.fillna("").str.contains(pattern_escalade, na=False)
        &(t.ordre_message_conv>t.first_user_message)
    )
    
    
    #bot demande feedback 
    
    pattern_feedback = re.compile(
    r"""
    (?:
        # Français
        cela\s+vous\s+a[-\s]?t[-\s]?il\s+aid
        |
        est[-\s]?ce\s+que\s+cela\s+vous\s+a\s+aid
        |
        # Anglais
        did\s+this\s+help
        |
        did\s+that\s+help
        |
        was\s+this\s+helpful
        |
        did\s+this\s+resolve
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
    )
    
    t['bot_asked_feedback']=(t.speaker.apply(classify_speaker) =='bot')&(t.message.str.contains(pattern_feedback, na=False))&(t.ordre_message_conv>t.first_bot_message)
    
    #bot demande clarification 
    
    pattern_bot_clarification = re.compile(
    r"""
    (
        afin\s+de\s+mieux\s+vous\s+aider
        | pour\s+mieux\s+vous\s+aider
        | pourriez[-\s]?vous
        | pouvez[-\s]?vous
        | j['’]?aurais\s+besoin\s+de\s+plus\s+d['’]?informations
        | need\s+(?:some\s+)?more\s+information
        | could\s+you\s+(?:please\s+)?(?:provide|clarify|confirm|share|send)
        | can\s+you\s+(?:please\s+)?(?:provide|clarify|confirm|share|send)
        | to\s+better\s+assist\s+you
        | in\s+order\s+to\s+better\s+assist
        | to\s+make\s+sure
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )

    t["bot_asks_clarification"] = (
        (t["speaker_type"] == "bot")
        & t["message"].fillna("").str.contains(pattern_bot_clarification, regex=True)
    )
    
    #bot annonce escalade en cours
    
    pattern_escalade_in_progress = re.compile(
    r"""
    (
        # FR
        vous\s+serez\s+bient[oô]t\s+
        (?:
            mis\s+en\s+relation
            | connect[eé]
            | transf[eé]r[eé]
        )
        .*?
        (?:
            expert
            | agent
            | humain
            | heartist
            | [ée]quipe
        )

        |

        # EN
        you\s+will\s+soon\s+be\s+
        (?:
            connected
            | transferred
            | put\s+in\s+touch
        )
        .*?
        (?:
            expert
            | agent
            | human
            | heartist
            | support\s+team
            | team
        )

        |

        # Variantes sans "soon"
        you\s+will\s+be\s+
        (?:
            connected
            | transferred
        )
        .*?
        (?:
            expert
            | agent
            | human
            | heartist
        )

        |

        vous\s+serez\s+
        (?:
            mis\s+en\s+relation
            | connect[eé]
            | transf[eé]r[eé]
        )
        .*?
        (?:
            expert
            | agent
            | humain
            | heartist
        )
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )
    
    t["bot_escalade_in_progress"] = (
    (t["speaker_type"] == "bot")
    & t["message"].fillna("").str.contains(pattern_escalade_in_progress, regex=True)
    )
    
    #escalation_context
    
    escal_confirmed = (
        t[t["bot_escalade_in_progress"] == True]
        .groupby("id")["ordre_message_conv"]
        .min()
        .reset_index()
        .rename(columns={"ordre_message_conv": "escal_confirmed_msg"})
    )

    t = t.merge(escal_confirmed, how="left", on="id")
    
    t['escalation_in_process']=(
        (t['ordre_message_conv']> t["escal_confirmed_msg"])
        &(~ t["human_agent_has_spoken_before"])
    )
    
    
    
    
    #user pose question followup
    
    pattern_user_escalade_followup = re.compile(
    r"""
    (
        ^\s*(when|quand)\s*\??\s*$
        | do\s+you\s+need
        | can\s+you
        | can\s+i
        | could\s+you
        | should\s+i\s+(?:send|give|provide)
        | before\s+connecting
        | before\s+you\s+connect
        | heartist\s+expert
        | reference\s+(?:first|number)?
        | resaweb
        | avant\s+de\s+(?:me\s+)?mettre\s+en\s+relation
        | avez[-\s]?vous\s+besoin
        | dois[-\s]?je\s+(?:envoyer|donner|fournir)
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )
    
    t["user_escalade_followup"] = (
    (t["speaker_type"] == "user")
    & t['escalation_in_process']
    & t["message"].fillna("").str.contains(pattern_user_escalade_followup, regex=True)
    )
    
    # 4. Requête explicite de nouvelle question
    pattern_new_question_explicit = re.compile(
        r"""
        (
            can\s+i\s+ask
            | i\s+have\s+another\s+question
            | another\s+question
            | one\s+more\s+question
            | i\s+need\s+help\s+with\s+something\s+else
            | another\s+issue
            | different\s+question
            | j['’]ai\s+une\s+autre\s+question
            | autre\s+question
            | je\s+voudrais\s+demander\s+autre\s+chose
            | puis[-\s]?je\s+poser\s+une\s+autre\s+question
            | j['’]ai\s+un\s+autre\s+sujet
            | autre\s+sujet
            | autre\s+probl[eè]me
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )

    t["user_explicit_new_question"] = (
        (t["speaker_type"] == "user")
        & t["message_clean"].str.contains(pattern_new_question_explicit, na=False)
    )

    
    #User pose une question au bot
    
    pattern_user_clarification = re.compile(
    r"""
    (
        # ENGLISH
        
        ^\s*by\s+.+\s+you\s+mean\b
        | ^\s*you\s+mean\b
        | ^\s*it\s+means\b
        | ^\s*that\s+means\b
        | ^\s*so\s+you\s+mean\b
        | ^\s*so\s+it\s+means\b
        | ^\s*do\s+you\s+mean\b
        | ^\s*are\s+you\s+saying\b
        | ^\s*if\s+i\s+understand\s+correctly\b
        | ^\s*if\s+i\s+understand\b
        | ^\s*you\s+are\s+saying\b
        
        # FRENCH
        
        | ^\s*tu\s+veux\s+dire\b
        | ^\s*vous\s+voulez\s+dire\b
        | ^\s*ça\s+veut\s+dire\b
        | ^\s*cela\s+veut\s+dire\b
        | ^\s*donc\s+ça\s+veut\s+dire\b
        | ^\s*donc\s+cela\s+veut\s+dire\b
        | ^\s*donc\s+vous\s+voulez\s+dire\b
        | ^\s*donc\s+tu\s+veux\s+dire\b
        | ^\s*si\s+je\s+comprends\b
        | ^\s*si\s+j['’]ai\s+bien\s+compris\b
        | ^\s*autrement\s+dit\b
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )
    
    t["user_clarification_only"] = (
    (t["speaker_type"] == "user")
    & t["message"].fillna("").str.contains(pattern_user_clarification)
    )
    
    #lien/attechment 
    
    pattern_attachment_or_link = re.compile(
    r"""
    (
        ^\s*attachment\s+\d+\s*:
        | https?://
        | www\.
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )
    
    t["user_attachment_or_link_only"] = (
    (t["speaker_type"] == "user")
    & t["message"].fillna("").str.contains(pattern_attachment_or_link)
    )
    
    #continuation 
    
    pattern_continuation = re.compile(
    r"""
    (
        ^\s*je\s+n['’]ai\s+pas\s+fini\b
        | ^\s*i\s+(am|was)\s+not\s+finished\b
        | ^\s*wait\b
        | ^\s*attendez\b
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )
    
    t["user_continuation_only"] = (
    (t["speaker_type"] == "user")
    & t["message"].fillna("").str.contains(pattern_continuation)
    )
    
    #demande escalade
    
    pattern_handoff_or_ticket_only = re.compile(
    r"""
    (
        # FR — ticket
        (créer?|ouvrir|faire|logg?er)\s+(?:un|le|mon|ce)?\s*(?:ticket|dossier)
        
        |
        
        # FR — agent / humain / expert
        (parler|échanger|discuter|être\s+mis\s+en\s+relation|me\s+mettre\s+en\s+relation|me\s+connecter)
        \s+(avec\s+)?(un\s+)?(agent|humain|conseiller|expert|heartist)
        
        |
        
        # EN — ticket / case
        (open|create|raise|log|submit)\s+(?:a|the|my|this)?\s*(?:ticket|case)
        
        |
        
        # EN — agent / human / expert
        (talk|speak|chat|connect(\s+me)?|put\s+me\s+in\s+touch)
        \s+(to|with)?\s*(a\s+)?(human|agent|advisor|expert|heartist)
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )
    
    t["user_handoff_or_ticket_only"] = (
    (t["speaker_type"] == "user")
    & t["message"].fillna("").str.contains(pattern_handoff_or_ticket_only)
    )
    
    #acceptation escalade
    
    pattern_acceptation = re.compile(
    r"""
    (
        \b(oui|yes|yep|yeah|ok|okay)\b
        |
        \b(d['’]?accord|volontiers|bien sûr|bien sur)\b
        |
        \b(je\s+veux\s+bien|allez-y|vas-y|go\s+ahead|please|yes\s+please)\b
        |
        \b(connectez[-\s]?moi|mettez[-\s]?moi\s+en\s+relation|transférez[-\s]?moi)\b
        |
        \b(connect\s+me|transfer\s+me|put\s+me\s+through)\b
        |
        parler\s+à\s+(un\s+)?(expert|agent|humain)
        |
        talk\s+to\s+(an?\s+)?(agent|human|expert)
        |
        \b(agent|expert|humain|human)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )
    
    t['user_accepts'] = (
    (t['speaker_type']=='user')&(t["message"].fillna("").str.contains(pattern_acceptation))
    )
    
    
    #demande dupdate sujet existant 
    
    pattern_update_request = re.compile(
    r"""
    (
        # FR
        update
        | nouvelles?
        | statut
        | avancement
        | suivi
        | retour
        | des?\s+nouvelles
        | où\s+en\s+est
        | qu['’]en\s+est[- ]il
        | vérifier\s+(mon\s+)?(ticket|cas|dossier)
        | concernant\s+(mon\s+)?(ticket|cas|dossier)

        |

        # EN
        any\s+update
        | status
        | follow[- ]?up
        | progress
        | news\s+about
        | check\s+(my\s+)?(ticket|case)
        | update\s+on
        | regarding\s+(my\s+)?(ticket|case)
    )
    """,
    re.IGNORECASE | re.VERBOSE
    )
    
    t["user_is_update_request"] = (
        (t.speaker.apply(classify_speaker) == 'user')
        & t["message"].fillna("").str.contains(pattern_update_request, regex=True)
    )
    
    
    # tracking message user
    
    t["user_msg_counter"] = (
        (t["speaker_type"] == "user")
        .groupby(t["id"])
        .cumsum()
    )
    

    t["between_user_block"] = t["user_msg_counter"]

    t["bot_asked_since_last_user"] = (
        t.groupby(["id", "between_user_block"])["bot_asks_anything_else"]
        .cummax()
        .shift(fill_value=False)
    )

    t["bot_escalated_since_last_user"] = (
        t.groupby(["id", "between_user_block"])["bot_proposed_escalation"]
        .cummax()
        .shift(fill_value=False)
    )
    
    
    t["bot_asked_fb_since_last_user"] = (
        t.groupby(["id", "between_user_block"])["bot_asked_feedback"]
        .cummax()
        .shift(fill_value=False)
    )
    
    t["bot_asked_clarification_since_last_user"] = (
        t.groupby(["id", "between_user_block"])["bot_asks_clarification"]
        .cummax()
        .shift(fill_value=False)
    )
    
    
    # Sécurité : uniquement pertinent sur les messages user
    t["bot_asked_since_last_user"] = (
        t["bot_asked_since_last_user"] & (t["speaker_type"] == "user")
    )

    t["bot_escalated_since_last_user"] = (
        t["bot_escalated_since_last_user"] & (t["speaker_type"] == "user")
    )
    
    t["bot_asked_fb_since_last_user"] = (
        t["bot_asked_fb_since_last_user"] & (t["speaker_type"] == "user")
    )
    
    t["bot_asked_clarification_since_last_user"] = (
        t["bot_asked_clarification_since_last_user"] & (t["speaker_type"] == "user")
    )
        
    t["user_accepts_escalation"]= (
        (t["bot_escalated_since_last_user"]) & (t['user_accepts'])
    )
    
    # 7. Définition finale nouvelle question
    t["is_new_question_start"] = (
        (t["speaker_type"] == "user")
        & t["has_user_message"]
        & (
            # premier message user hors greetings
            #(
                #(t["ordre_message_conv"] == t["first_user_message"])
                #&(~(t['user_is_update_request']))
                #&(~(t["user_handoff_or_ticket_only"]))
            #)
            #|
            # nouvelle question explicite (hors mise en relation humain)
            (
                t["user_explicit_new_question"]
                &(~t['human_agent_has_spoken_before'])
            )
            |
            # requête 
            # ET
            # (bot demandé si autre chose) OU (bot a proposé escalade) OU (bot a demandé feedback)
            # ET
            # (nest pas un followup d'escalade)
            # ET
            # (nest pas une suite de demande de clarification)
            # ET
            # l'agent humain n'a pas encore intervenu
            (
                t["user_message_is_request"]
                & (
                    t["bot_asked_since_last_user"]
                    | t["bot_escalated_since_last_user"]
                    | t["bot_asked_fb_since_last_user"]
                    | (t["ordre_message_conv"] == t["first_user_message"])
                )
                
                &(~t['human_agent_has_spoken_before'])
                
                &(~t['bot_asked_clarification_since_last_user'])
                
                &(~(t["user_escalade_followup"]))
                
                &(~(t['user_clarification_only']))
                
                &(~(t["user_handoff_or_ticket_only"]))
                
                &(~(t["user_continuation_only"]))
                
                &(~(t["user_attachment_or_link_only"]))
                
                &(~(t["user_accepts_escalation"]))
                
                &(~(t['user_is_update_request']))
            )
        )
    )

    # 8. Ranking questions
    t["question_rank"] = (
        t.groupby("id")["is_new_question_start"]
        .cumsum()
    )

    # Messages avant le premier user rattachés à Q1 si user existe
    t.loc[
        (t["has_user_message"]) & (t["question_rank"] == 0),
        "question_rank"
    ] = 1

    # Si aucun user message : Q0
    t.loc[
        ~t["has_user_message"],
        "question_rank"
    ] = 0

    t["question_id"] = np.where(
        t["question_rank"] > 0,
        t["id"].astype(str) + "_Q" + t["question_rank"].astype(int).astype(str),
        np.nan
    )

    t["ordre_message_question"] = (
        t.groupby(["id", "question_id"])
        .cumcount() + 1
    )

    return t

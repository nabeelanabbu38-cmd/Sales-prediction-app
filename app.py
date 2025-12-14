{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNfKVGDC09emJDEwVFAGzQX",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/nabeelanabbu38-cmd/Sales-prediction-app/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Q_KSW_QQ71Xe",
        "outputId": "cd6cfc29-8020-46ff-e856-63dfada74edc",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "import PyPDF2\n",
        "import nltk\n",
        "import re\n",
        "\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "\n",
        "nltk.download('stopwords')\n",
        "from nltk.corpus import stopwords\n",
        "\n",
        "def extract_text_from_pdf(pdf_file):\n",
        "    text = \"\"\n",
        "    reader = PyPDF2.PdfReader(pdf_file)\n",
        "    for page in reader.pages:\n",
        "        text += page.extract_text()\n",
        "    return text\n",
        "\n",
        "def clean_text(text):\n",
        "    text = text.lower()\n",
        "    text = re.sub(r'[^a-zA-Z ]', '', text)\n",
        "    text = text.split()\n",
        "    text = [word for word in text if word not in stopwords.words('english')]\n",
        "    return \" \".join(text)\n",
        "\n",
        "def calculate_similarity(resume_texts, jd_text):\n",
        "    documents = resume_texts + [jd_text]\n",
        "    vectorizer = TfidfVectorizer()\n",
        "    tfidf_matrix = vectorizer.fit_transform(documents)\n",
        "    similarity_scores = cosine_similarity(\n",
        "        tfidf_matrix[:-1], tfidf_matrix[-1]\n",
        "    )\n",
        "    return similarity_scores.flatten()\n",
        "\n",
        "st.set_page_config(page_title=\"AI Resume Screening System\")\n",
        "\n",
        "st.title(\"🧠 AI Resume Screening System\")\n",
        "\n",
        "uploaded_files = st.file_uploader(\n",
        "    \"Upload Resume PDFs\",\n",
        "    type=[\"pdf\"],\n",
        "    accept_multiple_files=True\n",
        ")\n",
        "\n",
        "job_description = st.text_area(\"Paste Job Description\")\n",
        "\n",
        "if st.button(\"🔍 Screen Resumes\"):\n",
        "    if not uploaded_files or job_description.strip() == \"\":\n",
        "        st.warning(\"Please upload resumes and enter job description\")\n",
        "    else:\n",
        "        resumes = []\n",
        "        resume_names = []\n",
        "\n",
        "        for file in uploaded_files:\n",
        "            text = extract_text_from_pdf(file)\n",
        "            cleaned_text = clean_text(text)\n",
        "            resumes.append(cleaned_text)\n",
        "            resume_names.append(file.name)\n",
        "\n",
        "        cleaned_jd = clean_text(job_description)\n",
        "        scores = calculate_similarity(resumes, cleaned_jd)\n",
        "\n",
        "        results = sorted(\n",
        "            zip(resume_names, scores),\n",
        "            key=lambda x: x[1],\n",
        "            reverse=True\n",
        "        )\n",
        "\n",
        "        st.subheader(\"📊 Resume Match Results\")\n",
        "        for name, score in results:\n",
        "            st.write(f\"📄 {name} — **{round(score*100,2)}% match**\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qecScLS--MZk",
        "outputId": "6786839c-dc26-42d6-a4a1-b697729a75a2"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "app.py\tsample_data\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YRgWNZnp-URf",
        "outputId": "963824fa-955d-44c7-e594-89b8ddd43f49"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.52.1-py3-none-any.whl.metadata (9.8 kB)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.5.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<7,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.2.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (8.3.1)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.0.2)\n",
            "Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (25.0)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<13,>=7.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (11.3.0)\n",
            "Requirement already satisfied: protobuf<7,>=3.20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.29.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.32.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (9.1.2)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (4.15.0)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.12/dist-packages (from streamlit) (3.1.45)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
            "Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.5.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (4.25.1)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2.13.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.12/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.4.4)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.11)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2025.11.12)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.0.3)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (25.4.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2025.9.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.37.0)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.30.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n",
            "Downloading streamlit-1.52.1-py3-none-any.whl (9.0 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m9.0/9.0 MB\u001b[0m \u001b[31m26.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m45.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pydeck, streamlit\n",
            "Successfully installed pydeck-0.9.1 streamlit-1.52.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pkill streamlit"
      ],
      "metadata": {
        "id": "6H9Wfe_P-kAt"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!apt-get update\n",
        "!apt-get install -y cloudflared"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IT7w9iJG_Qfb",
        "outputId": "8d706c87-3af9-4b3b-cfd0-df0efa6f1d06"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\r0% [Working]\r            \rHit:1 https://cli.github.com/packages stable InRelease\n",
            "\r0% [Connecting to archive.ubuntu.com (91.189.92.24)] [Connecting to security.ub\r                                                                               \rGet:2 https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/ InRelease [3,632 B]\n",
            "Get:3 https://r2u.stat.illinois.edu/ubuntu jammy InRelease [6,555 B]\n",
            "Get:4 http://security.ubuntu.com/ubuntu jammy-security InRelease [129 kB]\n",
            "Hit:5 http://archive.ubuntu.com/ubuntu jammy InRelease\n",
            "Get:6 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]\n",
            "Hit:7 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease\n",
            "Hit:8 https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu jammy InRelease\n",
            "Get:9 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [127 kB]\n",
            "Get:10 https://r2u.stat.illinois.edu/ubuntu jammy/main amd64 Packages [2,849 kB]\n",
            "Get:11 https://r2u.stat.illinois.edu/ubuntu jammy/main all Packages [9,537 kB]\n",
            "Get:12 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [3,623 kB]\n",
            "Get:13 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [3,961 kB]\n",
            "Get:14 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1,598 kB]\n",
            "Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [6,410 kB]\n",
            "Fetched 28.4 MB in 6s (4,691 kB/s)\n",
            "Reading package lists... Done\n",
            "W: Skipping acquire of configured file 'main/source/Sources' as repository 'https://r2u.stat.illinois.edu/ubuntu jammy InRelease' does not seem to provide it (sources.list entry misspelt?)\n",
            "Reading package lists... Done\n",
            "Building dependency tree... Done\n",
            "Reading state information... Done\n",
            "E: Unable to locate package cloudflared\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ftx_4bCV_ZmK",
        "outputId": "dc903c1f-e65b-462d-d2f7-2d3b70a271a5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  URL: \u001b[0m\u001b[1mhttp://0.0.0.0:8501\u001b[0m\n",
            "\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!cloudflared tunnel --url http://localhost:8501"
      ],
      "metadata": {
        "id": "I7kTX8luALgx"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
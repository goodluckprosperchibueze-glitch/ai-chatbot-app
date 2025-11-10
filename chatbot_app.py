# -------------------------
# MAIN CHAT UI (left column) - ENHANCED
# -------------------------

with col_left:
    st.markdown("### Chat with PopKing AI")
    st.markdown("Tips: Use **modes** to change behavior. Wikipedia lookup triggers on 'who is', 'what is', etc.")

    # Dynamic history container
    history_placeholder = st.empty()

    # Input area
    cols = st.columns([4, 1, 1])
    user_text = cols[0].text_input("Type message here...", key="user_input", label_visibility="collapsed")
    voice_file = cols[1].file_uploader(
        "Upload voice (wav/mp3, optional)", type=["wav", "mp3"], key="voice_upload", label_visibility="collapsed"
    )
    send_btn = cols[2].button("Send")

    # Display current history
    def display_history():
        with history_placeholder.container():
            for msg in st.session_state.history:
                if msg["role"] == "user":
                    st.markdown(f"<div class='msg_user'>**You:** {msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='msg_assistant'>**{msg.get('role_label','PopKing AI')}:** {msg['content']}</div>",
                        unsafe_allow_html=True,
                    )

    display_history()  # initial render

    # Image generation expander
    with st.expander("🖼️ Image generation (optional)"):
        img_prompt = st.text_area("Describe the image you want", value="", height=80, key="img_prompt")
        img_style = st.selectbox("Style (suggestion)", ["photorealistic", "digital art", "anime", "cartoon", "fantasy"], index=0)
        if st.button("Generate Image", key="gen_img_btn"):
            if not HF_TOKEN:
                st.error("Image generation needs a Hugging Face token. Set it in the right panel.")
            else:
                with st.spinner("Generating image..."):
                    try:
                        full_prompt = f"{img_prompt} -- style: {img_style}"
                        img_bytes = hf_generate_image_bytes(full_prompt, HF_TOKEN)
                        st.image(img_bytes)
                        st.success("Image generated.")
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

    # Wikipedia quick search expander
    with st.expander("🔎 Knowledge / Quick search (Wikipedia)"):
        search_q = st.text_input("Ask something to lookup (Wikipedia)", value="", key="wiki_q")
        if st.button("Lookup Wikipedia", key="wiki_lookup_btn"):
            if search_q.strip():
                with st.spinner("Searching Wikipedia..."):
                    summ = wiki_lookup(search_q, sentences=3)
                    if summ:
                        st.markdown(f"**Wikipedia summary:**\n\n{safe_clean(summ)}")
                    else:
                        st.info("No summary found. Try rephrasing or use fewer words.")

    # -------------------------
    # HANDLE sending message
    # -------------------------
    user_message_final = None

    # Voice file takes priority
    if voice_file is not None:
        if sr is None:
            st.warning("Speech recognition not installed; please type or install `speechrecognition`.")
        else:
            with st.spinner("Recognizing speech..."):
                recognized = recognize_speech_from_file(voice_file)
                if recognized:
                    user_message_final = recognized
                    st.success(f"Recognized: {recognized}")
                else:
                    st.error("Could not recognize speech. Try clearer audio or type the message.")
    elif send_btn and user_text and user_text.strip():
        user_message_final = user_text.strip()

    # If we have a message to send:
    if user_message_final:
        # Save user message
        st.session_state.history.append({"role": "user", "content": user_message_final})
        display_history()  # update history immediately

        # Quick heuristic for Wikipedia facts
        quick_fact = None
        if enable_wiki and re.search(r"\b(who is|what is|when is|where is|tell me about)\b", user_message_final, re.IGNORECASE):
            quick_fact = wiki_lookup(user_message_final, sentences=2)

        with st.spinner("PopKing AI is thinking..."):
            try:
                # Mode-aware response
                current_mode = st.session_state.settings.get("mode", "Chat (default)")

                if quick_fact and current_mode == "Chat (default)":
                    ai_reply = quick_fact + "\n\n(Quick knowledge summary from Wikipedia.)"
                else:
                    if current_mode == "Story Mode":
                        story_prompt = (
                            f"**INSTRUCTION: Ignore previous conversation history for this turn.**\n"
                            f"{st.session_state.persona} - Write a **detailed, imaginative, and lively story**, at least 3 paragraphs, "
                            f"based on the user's request: {user_message_final}\n\nStart the story now, only respond with narrative."
                        )
                        max_len_story = max(512, max_len * 2)
                        ai_reply = generate_response_with_transformers(story_prompt, temp, top_p, max_len_story)

                    elif current_mode == "Deep Search":
                        deep_prompt = (
                            f"**INSTRUCTION: Ignore previous conversation history for this turn.**\n"
                            f"{st.session_state.persona} - Provide a **well-structured, factual, numbered/bullet-point answer**. "
                            f"If uncertain about any detail, say 'I am not certain about this detail' and suggest search keywords. "
                            f"Question: {user_message_final}\n\nAnswer:"
                        )
                        ai_reply = generate_response_with_transformers(deep_prompt, temp, top_p, max_len)

                    else:
                        # Default Chat
                        ai_reply = generate_response(user_message_final, temperature=temp, top_p_val=top_p, max_length_val=max_len)

                ai_reply = safe_clean(ai_reply) or "Sorry, I couldn't produce an answer. Try rephrasing."

                # Save AI reply and update display
                st.session_state.history.append({"role": "assistant", "role_label": "PopKing AI", "content": ai_reply})
                display_history()

                # TTS (optional)
                if enable_voice and gTTS is not None:
                    audio_bytes = text_to_speech_bytes(ai_reply)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

            except Exception as e:
                st.error(f"AI generation error: {e}")

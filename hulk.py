import streamlit as st
from streamlit.components.v1 import html

# Create a container for the footer
footer_container = st.container()

# Add interactive SVG microphone icon
svg_code = """
<div id="mic-container" style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); cursor: pointer;">
    <lottie-player id="mic-icon" src="https://assets10.lottiefiles.com/packages/lf20_5ttkzjhr.json" background="transparent" speed="1" style="width: 48px; height: 48px;" loop autoplay></lottie-player>
</div>
<script>
    document.getElementById('mic-container').addEventListener('click', function() {
        alert('Microphone icon clicked!');
    });
</script>
"""
html(svg_code)

# Display a simple message
st.write("Click the microphone icon to test JavaScript functionality.")
 
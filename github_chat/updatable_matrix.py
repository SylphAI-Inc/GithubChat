import streamlit as st

# Iteratively update the state via buttons.
# for each new conversation response,
# 1) create a next button
# 2) crate a prev button
# 3) On each button, add an event listener, where the buttons follow the logic listed in steps 1 and 2

# Main problems:
# 1) How do we know which button is clicked and when?
# 2) When a button is clicked, it needs to know which indices to update.
# 3) Need a mechanism for adding "rows", columns are fixed at k

print(st.session_state)

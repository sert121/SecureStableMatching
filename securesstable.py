
import streamlit as st
import pandas as pd 
import numpy as np
import time
import altair as alt
import sys
import random
from phe import paillier
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def algo(n, men_preferences, women_preferences):
    public_key, private_key = paillier.generate_paillier_keypair()
    # n = int(sys.argv[1])
    # print(f'{n} clients')
    ##### SETUP ############
    ####### CRYPTOGRAPHIC PRIMITIVES ##########

    def EQ_TEST(x, y, private_key):
        a = x - y
        if private_key.decrypt(a) == 0:
            return 1
        else:
            return 0

    def INDEX(a, ciphertext, private_key):
        index = 0
        while True:
            if EQ_TEST(a[index], ciphertext, private_key) == 1:
                break
            index += 1
        return index

    def COMPARE(c1, c2, private_key, n):
        D = []
        for i in range(1, n):
            D_i = c1 - c2 - public_key.encrypt(i)
            D.append(D_i)

        D = [re_encrypt(i) for i in D]
        sigma = np.random.permutation(len(D))
        D = [D[i] for i in sigma]
        for i in range(len(D)):
            if EQ_TEST(D[i], public_key.encrypt(0), private_key) == 1:
                return True

        return False

    def re_encrypt(c1):
        temp = public_key.encrypt(0)
        temp = c1 + temp
        return temp

    def add_1(c1):
        temp = public_key.encrypt(1)
        temp = c1 + temp
        return temp
    ##### INPUT SUBMISSION ############

    # def gen_prefs(n):
    #     l = np.empty([n, n]).astype(int).tolist()
    #     for i in range(n):
    #         pi = np.random.permutation(n).astype(int).tolist()
    #         l[i] = [x for x in pi]
    #     return l

    # men_preferences = gen_prefs(n)
    # women_preferences = gen_prefs(n)
    # print(men_preferences)
    # print(women_preferences)

    def gen_enc(n, preferences):
        men = dict()
        for i in range(n):
            men[i] = [public_key.encrypt(x) for x in preferences[i]]
        return men

    men = gen_enc(n, men_preferences)
    women = gen_enc(n, women_preferences)

    # ##### INPUT SUBMISSION ############

    # #### ADDITION OF FAKE MEN ########
    # # The matching authorities define an additional n fake men
    for i in range(n, 2*n):
        pi = np.random.permutation(n).tolist()
        men[i] = [public_key.encrypt(x) for x in pi]

    # # augmented preferences for woman is (i-1) for i in (n+1,2n)
    augment = list(range(n, 2*n))
    augment_ranklist = [public_key.encrypt(x) for x in augment]
    for i in range(0, n):
        women[i].extend(augment_ranklist)

    # #### ADDITION OF FAKE MEN ########

    # ######## BID CREATION #########
    bids = dict()

    def create_bids():
        a = list(range(1, n+1))

        for i in range(2*n):
            women_pref = []
            for j in range(n):
                women_pref.append(women[j][i])

            bids[i] = np.array([public_key.encrypt(i+1), np.array(men[i]), np.array([public_key.encrypt(x)
                                                                                     for x in a]), np.array(women_pref), public_key.encrypt(0)])  # each element of dict is a nparray

    create_bids()
    Free_bids, Engaged_bids = [], []
    for i in range(n):
        Free_bids.append(bids[i])
        Engaged_bids.append(
            np.array([bids[i+n], public_key.encrypt(i+1), women[i][i+n]]))

    Free_bids = np.array(Free_bids)
    Engaged_bids = np.array(Engaged_bids)

    # ######## BID CREATION #########

    ###### Initial Mixing #########
    pi = np.random.permutation(n)

    def pi_per(ciphers):
        ciphers = np.apply_along_axis(lambda x: re_encrypt(x), 0, ciphers)
        ciphers = ciphers[pi]
        return ciphers

    def External_Mix_Free(bids):
        if len(bids) == 0:
            return bids

        for i in range(len(bids)):
            bids[i] = np.array([re_encrypt(bids[i][0]), np.array(np.apply_along_axis(lambda x: re_encrypt(x), 0, bids[i][1])), np.array(
                np.apply_along_axis(lambda x: re_encrypt(x), 0, bids[i][2])), np.array(np.apply_along_axis(lambda x: re_encrypt(x), 0, bids[i][3])), re_encrypt(bids[i][4])])

        sigma = np.random.permutation(len(bids))
        bids = bids[sigma]
        return bids

    def Internal_Mix_Free(bids):
        if len(bids) == 0:
            return bids

        for i in range(len(bids)):
            bids[i] = np.array([bids[i][0], pi_per(bids[i][1]), pi_per(
                bids[i][2]), pi_per(bids[i][3]), bids[i][4]])
        return bids

    def External_Mix_Engaged(bids):
        if len(bids) == 0:
            return bids

        for i in range(len(bids)):
            bids[i] = np.array([np.array([re_encrypt(bids[i][0][0]), np.array(np.apply_along_axis(lambda x: re_encrypt(x), 0, bids[i][0][1])), np.array(np.apply_along_axis(
                lambda x: re_encrypt(x), 0, bids[i][0][2])), np.array(np.apply_along_axis(lambda x: re_encrypt(x), 0, bids[i][0][3])), re_encrypt(bids[i][0][4])]), re_encrypt(bids[i][1]), re_encrypt(bids[i][2])])

        sigma = np.random.permutation(len(bids))
        bids = bids[sigma]
        return bids

    def Internal_Mix_Engaged(bids):
        if len(bids) == 0:
            return bids

        for i in range(len(bids)):
            bids[i] = np.array([np.array([bids[i][0][0], pi_per(bids[i][0][1]), pi_per(
                bids[i][0][2]), pi_per(bids[i][0][3]), bids[i][0][4]]), bids[i][1], bids[i][2]])
        return bids

    ###### Initial Mixing #########
    Free_bids = External_Mix_Free(Free_bids)
    Engaged_bids = External_Mix_Engaged(Engaged_bids)
    Free_bids = Internal_Mix_Free(Free_bids)
    Engaged_bids = Internal_Mix_Engaged(Engaged_bids)

    FREE_BIDS = {1: Free_bids}
    ENGAGED_BIDS = {1: Engaged_bids}

    ######### Computing a stable match ###########
    for i in range(2, n+2):
        FREE_BIDS[i] = None
        ENGAGED_BIDS[i] = None
    ###########################

    for k in range(1, n+1):
        while len(FREE_BIDS[k]) > 0:
            ind = random.randrange(0, len(FREE_BIDS[k]))
            bid = FREE_BIDS[k][ind]
            FREE_BIDS[k] = np.delete(FREE_BIDS[k], ind, 0)

            index = INDEX(bid[1], bid[4], private_key)
            E_j = bid[2][index]
            E_sji = bid[3][index]

            for i in range(len(ENGAGED_BIDS[k])):
                if EQ_TEST(E_j, ENGAGED_BIDS[k][i][1], private_key) == 1:
                    conflict_bid = ENGAGED_BIDS[k][i]
                    ENGAGED_BIDS[k] = np.delete(ENGAGED_BIDS[k], i, 0)
                    break

            new_engaged_bid = np.array([np.array(bid), E_j, E_sji])

            temp_Engaged_bids = np.array([new_engaged_bid, conflict_bid])
            # temp_Engaged_bids=External_Mix_Engaged(temp_Engaged_bids)
            result = COMPARE(
                temp_Engaged_bids[0][2], temp_Engaged_bids[1][2], private_key, 2*n)

            if not result:

                ENGAGED_BIDS[k] = np.append(
                    ENGAGED_BIDS[k], [temp_Engaged_bids[0]], axis=0)

                w1 = temp_Engaged_bids[1][0]
                w1[4] = add_1(w1[4])

                if FREE_BIDS[k+1] is None:
                    FREE_BIDS[k+1] = np.array([w1])
                else:
                    FREE_BIDS[k+1] = np.append(FREE_BIDS[k+1], [w1], axis=0)

            else:

                ENGAGED_BIDS[k] = np.append(
                    ENGAGED_BIDS[k], [temp_Engaged_bids[1]], axis=0)
                w1 = temp_Engaged_bids[0][0]
                w1[4] = add_1(w1[4])
                if FREE_BIDS[k+1] is None:
                    FREE_BIDS[k+1] = np.array([w1])

                else:
                    FREE_BIDS[k+1] = np.append(FREE_BIDS[k+1], [w1], axis=0)
            ####################

            ENGAGED_BIDS[k] = External_Mix_Engaged(ENGAGED_BIDS[k])
            ENGAGED_BIDS[k] = Internal_Mix_Engaged(ENGAGED_BIDS[k])
            FREE_BIDS[k] = Internal_Mix_Free(FREE_BIDS[k])
            FREE_BIDS[k+1] = Internal_Mix_Free(FREE_BIDS[k+1])

        ENGAGED_BIDS[k+1] = ENGAGED_BIDS[k]

        FREE_BIDS[k+1] = External_Mix_Free(FREE_BIDS[k+1])
        ENGAGED_BIDS[k+1] = External_Mix_Engaged(ENGAGED_BIDS[k+1])
        FREE_BIDS[k+1] = Internal_Mix_Free(FREE_BIDS[k+1])
        ENGAGED_BIDS[k+1] = Internal_Mix_Engaged(ENGAGED_BIDS[k+1])

    ######### Computing a stable match ###########

    ############# Decryption ###########
    pairs = []
    for bid in ENGAGED_BIDS[n+1]:
        pairs.append([bid[0][0], bid[1]])

    for pair in pairs:
        pair = [re_encrypt(pair[0]), re_encrypt(pair[1])]
    sigma = np.random.permutation(len(pairs))
    pairs = [pairs[i] for i in sigma]

    #########Decryption ################
    for i in range(len(pairs)):
        pairs[i] = [private_key.decrypt(
            pairs[i][0]), private_key.decrypt(pairs[i][1])]
    return pairs
# print(pairs)
st.set_page_config(layout="centered") 
st.title('Secure Stable Matching ')
n = st.slider('Enter the number the participants', 0, 50, 1)
st.write('Number of participants chosen: ',n)

participant_list_men = ['M' + str(i) for i in range(n)]
participant_list_women = ['W' + str(i) for i in range(n)]

complete_male_prefs = [] # stores the prefs of every male
complete_female_prefs = [] # stores prefs of every female
count,i =0,0

# genre = st.radio("Do you want to enter preferences for each person manually?",('No','Yes'))
add_selectbox = st.selectbox("How would you like to enter preferences?",("Enter Preferences List", "Upload Preferences List","Manually enter preferences"))
if add_selectbox =="Manually enter preferences":
    col1, col2 = st.beta_columns(2)
    with col2:
        st.write("Men Preferences")
        for i in participant_list_men:
            male_prefs = st.multiselect(f'{i} preferences',participant_list_women,[],key=i)
            complete_male_prefs.append(male_prefs)
            # print(male_prefs,type(male_prefs)) # stores it in a listformat [w1,w2,w3...] for each man
    with col1:
        st.write("Women Preferences")
        for j in participant_list_women:
            female_prefs = st.multiselect(f'{j} preferences',participant_list_men,[],key=j)
            complete_female_prefs.append(female_prefs)
            # print(female_prefs,type(female_prefs)) # stores it in listformat [m1,m2...] for each woman
    
    complete_male_prefs = [[int(j[1]) for j  in i] for i in complete_male_prefs] # this can be exported to main.py
    complete_female_prefs = [[int(j[1]) for j  in i] for i in complete_female_prefs] # this can be exported to main.py
    if st.button("Submit"):
        results = algo(n,complete_male_prefs,complete_female_prefs)
        if results!=None:
            print(results)
            st.write("hello")

elif add_selectbox=="Enter Preferences List":        
    mprefs = st.text_input('Men\'s Preferences', '')
    st.write('You\'ve entered: ', mprefs)
    wprefs = st.text_input('Women\'s Preferences', '')
    st.write('You\'ve entered: ', wprefs)    
else:
    uploaded_file = st.file_uploader("Upload your preferences")

# st.beta_container()
# st.beta_columns(spec)
col_01, col02 = st.beta_columns(2)
col_01.subheader('General info')
with st.beta_expander('Algorithm used'):
    st.markdown("[A private stable matching algorithm](https://crypto.stanford.edu/~pgolle/papers/stable.pdf)")

with st.beta_expander('Time Complexity'):

    st.write('Laptop specifications : i3-7th gen processor')
    x_vals_1 = [ i for i in range(3,21)]
    
    # Create a list of data to be  represented in y-axis
    y_vals_1 = [ 11.085 , 20.609 , 37.294 , 61.891 ,94.832 , 138.736 , 203.772, 281.717,380.379, 455.812, 572.636, 697.377,848.402,1017.834, 1220.885,1442.940, 1696.957,1980.268  ]
    chart_data = pd.DataFrame({"Number of clients":x_vals_1,"Time taken by algorithm (secs)": y_vals_1,"n3":[i**3 for i in range(3,21)]})
    data = chart_data.reset_index(drop=True).melt('Number of clients')

    data = alt.Chart(data).mark_line().encode(x='Number of clients',y='value',color='variable')
    
    st.altair_chart(data,use_container_width=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) #uncomment to remove made with streamlit option


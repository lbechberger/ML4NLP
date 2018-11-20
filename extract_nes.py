import nltk
from multiprocessing import Pool
import ks

txt = """
A progressive blog, Democrats.com, is offering a $1,000 reward to anyone who can get United States president George W. Bush to answer a question about pre-Iraq war intelligence. The contest comes after Bush has declined for a full month to answer any questions about a leaked secret British memo, which states, in reference to the Bush administration, that "the intelligence and facts were being fixed around the policy". Neither the US or UK government are disputing the document's accuracy. The $1,000 question is:
    In July 2002, did you and your administration "fix" the intelligence and facts about non-existent Iraqi WMD's and ties to terrorism - which were disputed by U.S. intelligence officials - to sell your decision to invade Iraq to Congress, the American people, and the world - as quoted in the Downing Street Minutes?
The contest will also reward anyone who asks the question to Mr. Bush with $100 dollars (video evidence is required). The reward money was generated from small donations to the website.
The contest comes during renewed criticism and pressure on the administration from citizens, members of congress, retired politicians, and constitutional lawyers, generated from the recent leak of the minutes and other British documents.
Earlier this month, White House spokesman Scott McClellan told reporters that the White House saw "no need" to respond to a letter from Congress asking questions about the memo. He also stated that "If anyone wants to know how the intelligence was used by the administration, all they have to do is go back and look at all the public comments over the course of the lead-up to the war in Iraq, and that’s all very public information. Everybody who was there could see how we used that intelligence."
One of the pre-war claims used by the Bush administration to justify the war was that Iraq possessed "weapons of mass destruction". After a thorough search, none were found. Senators on the Senate Intelligence Committee consider three possible causes for the apparent falsity of this claim:
    The WMD's were moved sometime between the war and the search for them.
    The intelligence produced by the CIA community was bad.
    The intelligence was satisfactory but was mis-handled by the Administration.
The last of these possibilities has not been formally investigated by the committee. """


# all_articles = [i[0] for i in ks.get_all_resource_category_mappings(ks.top_level_category_names).items()]
# all_articles = all_articles[:10]
# # for art in all_articles:
# #     print(ks.run_files_query(art))
# with Pool(processes=4) as pool:
#     res = pool.map(ks.run_files_query, all_articles)
#     print(print(res.get()))

recognized = []
sentences = nltk.sent_tokenize(txt)
for sent in sentences:
    word_tokenized = nltk.word_tokenize(sent)
    pos_tagged = nltk.pos_tag(word_tokenized)
    ne_chunked = nltk.ne_chunk(pos_tagged)
    recognized.append(" ".join([i[0] if isinstance(i, tuple) else "["+(" ".join([j[0] for j in i]))+"]" for i in ne_chunked]).replace(" ,",","))
    # print(ne_chunked)

print("\n".join(recognized))
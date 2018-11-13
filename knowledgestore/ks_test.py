# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:53:32 2018

@author: lbechberger
"""

import unittest
import ks

class TestKS(unittest.TestCase):

    # run_sparql_query
    def test_run_sparql_query_valid(self):
        expected_result = [{'s': 'http://en.wikinews.org/wiki/Bone_marrow_transplant_potentially_linked_to_cure_of_patient_with_AIDS#ev30'}, {'s': 'http://en.wikinews.org/wiki/China_responds_to_US_plan_for_import_quotas#ev30'}]
        actual_result = ks.run_sparql_query("SELECT ?s WHERE {?s rdf:type sem:Event} LIMIT 2")
        self.assertEquals(expected_result, actual_result)
    
    def test_run_sparql_query_valid_multiple_variables(self):
        expected_result = [{'a': 'http://dbpedia.org/resource/Leipzig_University', 'g': 'http://dbpedia.org/resource/Female'}]
        actual_result = ks.run_sparql_query("SELECT ?a ?g WHERE {dbpedia:Angela_Merkel dbo:almaMater ?a . dbpedia:Angela_Merkel dbo:gender ?g}")
        self.assertEquals(expected_result, actual_result)
    
    def test_run_sparql_query_invalid(self):
        expected_result = []
        actual_result = ks.run_sparql_query("SELECT ?s FROM {?s rdf:type sem:Event} LIMIT 10")
        self.assertEquals(expected_result, actual_result)

    # run_mention_query
    def test_run_mention_query_valid(self):
        expected_result = ['require']
        actual_result = ks.run_mention_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622", "nwr:pred")
        self.assertEquals(expected_result, actual_result)

    def test_run_mention_query_valid_multiple_results(self):
        expected_result = ['http://www.newsreader-project.eu/verbnet/order-60', 'http://www.newsreader-project.eu/verbnet/order-60-1', 'http://www.newsreader-project.eu/verbnet/require-103']
        actual_result = ks.run_mention_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622", "nwr:verbnetRef")
        self.assertEquals(expected_result, actual_result)

    def test_run_mention_query_invalid_property(self):
        expected_result = []
        actual_result = ks.run_mention_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622", "nwr:refersTo")
        self.assertEquals(expected_result, actual_result)

    def test_run_mention_query_invalid_uri(self):
        expected_result = []
        actual_result = ks.run_mention_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=0,629", "nwr:pred")
        self.assertEquals(expected_result, actual_result)

    def test_run_mention_query_valid_propBank(self):
        expected_result = ['http://www.newsreader-project.eu/propbank/issue.01']
        actual_result = ks.run_mention_query("http://en.wikinews.org/wiki/World_leaders_react_to_Obama_win#char=1881,1887", "nwr:propbankRef")
        self.assertEquals(expected_result, actual_result)   

    def test_run_mention_query_type_event(self):
        expected_result = ['ks:Mention', 'nwr:EntityMention', 'nwr:EventMention', 'nwr:TimeOrEventMention']
        actual_result = ks.run_mention_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622", "@type")
        self.assertEquals(expected_result, actual_result)

    def test_run_mention_query_type_relation(self):
        expected_result = ['ks:Mention', 'nwr:Participation', 'nwr:RelationMention']
        actual_result = ks.run_mention_query("http://en.wikinews.org/wiki/Angela_Merkel_elected_new_German_chancellor#char=563,585", "@type")
        self.assertEquals(expected_result, actual_result)

    # run_resource_query
    def test_run_resource_query_valid_hasMention(self):
        actual_result = ks.run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", "ks:hasMention")
        self.assertEquals(len(actual_result), 118)

    def test_run_resource_query_valid_title(self):
        expected_result = ['Mexican president defends emigration']
        actual_result = ks.run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", "dct:title")
        self.assertEquals(expected_result, actual_result)

    def test_run_resource_query_valid_created(self):
        expected_result = ['2005-05-14T02:00:00.000+02:00']
        actual_result = ks.run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", "dct:created")
        self.assertEquals(expected_result, actual_result)

    def test_run_resource_query_empty_property(self):
        expected_result = []
        actual_result = ks.run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", "dct:publisher")
        self.assertEquals(expected_result, actual_result)

    def test_run_resource_query_invalid_property(self):
        expected_result = []
        actual_result = ks.run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", "ks:refersTo")
        self.assertEquals(expected_result, actual_result)

    def test_run_resource_query_invalid_uri(self):
        expected_result = []
        actual_result = ks.run_resource_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration_NON_EXISTENT", "ks:hasMention")
        self.assertEquals(expected_result, actual_result)
  
    # run_files_query
    def test_run_files_query_valid(self):
        expected_result = "President of Mexico Vicente Fox spoke out on Friday in defense of Mexican workers headed North."
        actual_result = ks.run_files_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration")  
        self.assertEquals(expected_result, actual_result[:len(expected_result)])
        
    def test_run_files_query_invalid_resource(self):
        expected_result = ''
        actual_result = ks.run_files_query("http://en.wikinews.org/wiki/Mexican_president_defends_emigration_NON_EXISTENT")  
        self.assertEquals(expected_result, actual_result)
        
    # mention_URI_to_resource_URI
    def test_mention_URI_to_resource_URI_valid(self):
        expected_result = 'http://en.wikinews.org/wiki/Mexican_president_defends_emigration'
        actual_result = ks.mention_uri_to_resource_uri('http://en.wikinews.org/wiki/Mexican_president_defends_emigration#char=615,622')
        self.assertEquals(expected_result, actual_result)

    def test_mention_URI_to_resource_URI_invalid(self):
        expected_result = 'http://en.wikinews.org/wiki/Mexican_president_defends_emigration'
        actual_result = ks.mention_uri_to_resource_uri('http://en.wikinews.org/wiki/Mexican_president_defends_emigration')
        self.assertEquals(expected_result, actual_result)

    # get_applicable_news_categories
    def test_get_applicable_news_categories_match(self):
        expected_result = ['Politics and conflicts']
        actual_result = ks.get_applicable_news_categories("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", ks.top_level_category_names)
        self.assertEquals(expected_result, actual_result)

    def test_get_applicable_news_categories_mismatch(self):
        self.assertEquals(0, len(ks.get_applicable_news_categories("http://en.wikinews.org/wiki/Mexican_president_defends_emigration", ["Sports"])))
        
unittest.main()
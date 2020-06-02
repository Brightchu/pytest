#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:03:07 2020

@author: bright
"""

yes_votes = 42_572_654
no_votes = 43_132_495
percentage = yes_votes / (yes_votes + no_votes)
s='{:-9} YES votes  {:2.2%}'.format(yes_votes, percentage)
print(s)

hello = 'hello, world\n'
hellos = repr(hello)
print(hellos)
s = 'Hello, world.\n'
print(s)
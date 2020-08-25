#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Union, Tuple
import warnings
import ast
import re

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from utils import dataset_labels

#### Confusion Matrix Functions
def fill_cm(cm:pd.DataFrame) -> pd.DataFrame:
    """Fill confusion matrix with missing columns
    """
    # Check matrix is square
    all_cols = set(list(cm.columns)+list(cm.T.columns))
    for col in list(all_cols - set(cm.columns)):
        cm[col] = 0
    cm = cm.T # pivot - simpler way of adding data
    cm.sort_index(inplace=True) # order rows with new data
    for col in list(all_cols - set(cm.columns)):
        cm[col] = 0
    cm = cm.T
    cm.sort_index(inplace=True) # order rows with new data
    return cm

def display_cm(cm:np.array, labels:list, figsize:tuple=(20,20), savefig_loc:str=None):
    """Plot any confusion-matrix like array
    """
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='viridis', ax=ax)
    # save or return
    if savefig_loc is not None:
        plt.savefig(savefig_loc)
    return (fig, ax)

#### Analysis
class SPAAnalysis():
    def __init__(self, data_path:str, sheetname:Union[int, str]=0, **kwargs):
        """Loading data with default arguments:
            | speaker_col:str = 'speaker'
            | age_col:str = 'age_months'
            | file_col:str = 'file_id'
            | child_col:str = 'child'
            | tag_label:str = None
            | truth_col:str = 'y_true'
            | pred_col:str = 'y_pred'
            | index_col = 0
            | keep_default_na=False
            | na_values=''
        
        Input:
        -------
        sheetname: name or location of sheet containing data
        """
        # Dealing with kwargs
        default_kwargs = {
            'speaker_col': 'speaker',
            'age_col': 'age_months',
            'file_col': 'file_id',
            'child_col': 'child',
            'tag_col': None,
            'truth_col': 'y_true',
            'pred_col': 'y_pred',
            'index_col': 0,
            'keep_default_na':False,
            'na_values':''
        }
        for k,v in kwargs.items():
            if k in default_kwargs.keys():
                default_kwargs[k] = v
        # Setting main arguments
        for arg, value in default_kwargs.items():
            setattr(self, arg, value)

        if not os.path.isfile(data_path):
            raise ValueError(f"Path given '{data_path}' is not a valid path")
        else:
            try:
                format = data_path.split('.')[-1]
            except:
                raise ValueError(f"Path given has no extension; must be one of .csv|.tsv|.xlsx")
            self.data_path = data_path
            self.load_data(self.data_path, format, sheetname)
            # Checks
            for k in ['speaker_col','age_col','file_col','child_col']:
                if default_kwargs[k] not in self.data.columns:
                    warnings.warn(f"{k} column {default_kwargs[k]} not in data; replacing with None.")
                    setattr(self, k, None)
            file_tag = self.check_tag_pattern()
            if (file_tag is not None) and (file_tag in self.data.columns):
                setattr(self, 'tag_col', file_tag)



    ### Data
    def load_data(self, data_path, format, sheetname):
        if format == 'csv':
            self.data = pd.read_csv(data_path, keep_default_na=self.keep_default_na, na_values=self.na_values)
            self.sheetnames = None
        elif format == 'tsv':
            self.data = pd.read_csv(data_path, sep='\t', keep_default_na=self.keep_default_na, na_values=self.na_values)
            self.data.rename(columns = {col:col.lower() for col in self.data.columns}, inplace=True)
            self.sheetnames = None
        elif format == 'xlsx':
            self.dfs = pd.read_excel(data_path, None, index_col=self.index_col, keep_default_na=self.keep_default_na, na_values=self.na_values)
            self.sheetnames = list(self.dfs.keys())
            sheetname = sheetname if isinstance(sheetname, str) else self.sheetnames[sheetname]
            self.data = self.dfs[sheetname] 
        else:
            raise ValueError(f"Unsupported extension; must be one of .csv|.tsv|.xlsx")
    
    def check_tag_pattern(self):
        j = re.compile('spa_[0-9]{1}[a]{0,1}')
        pat = re.findall(j, self.data_path)
        if len(pat) > 0:
            return pat[0]
        return None

    def update_speakers(self, loc_replacement:dict = {'FAT':'MOT', 'INV':'MOT'}):
        self.data[self.speaker_col] = self.data[self.speaker_col].apply(lambda x: x if x not in loc_replacement else loc_replacement[x])

    def remove_ool(self, spa_col:str = None, inplace:bool = False):
        """Remove NAT/NOL/NEE tags from dataset for analysis; either inplace on data or returned
        """
        if (self.tag_col is None) and (spa_col is None):
            raise ValueError("'tag_col' is not set for the data; 'spa_col' must be given when calling the function, currently None")
        elif spa_col is None:
            spa_col = self.tag_col
        
        labels = dataset_labels(spa_col.upper(), add_empty_labels=False)
        if inplace:
            self.data = self.data[self.data[spa_col].isin(list(labels.keys()))]
        else:
            return self.data[self.data[spa_col].isin(list(labels.keys()))]

    ### Histograms over age / speakers
    def qt_spa_over_age(self, data:pd.DataFrame = None, spa_col:str = None, 
                        savefig_loc:str = None, figsize:Tuple[int,int] = (10,5), ax = None):
        """Group data by child and by age and count the number of labels used by each child at each age. 
        Data from other speakers must have been excluded from the data.
        """
        if (self.tag_col is None) and (spa_col is None):
            raise ValueError("'tag_col' is not set for the data; 'spa_col' must be given when calling the function, currently None")
        elif (spa_col is None):
            print(f"Using '{self.tag_col}' as 'spa_col'.")
            spa_col = self.tag_col
        if self.child_col is None:
            raise ValueError("'child_col' is not set, please specify")
        
        if data is None:
            print("Using class dataset as data (unfiltered).")
            data = self.data

        # Groupbys
        p = data.groupby(by=[self.age_col, self.child_col]).agg({
            spa_col: 'nunique'
        }).reset_index(drop=False).groupby(by=[self.age_col, spa_col]).agg({
            self.child_col: 'count'
        }).reset_index(drop=False).pivot_table(index=spa_col, columns=self.age_col, values=self.child_col).fillna(0)
        # Fill in index & fillna
        p = p.reindex(range(0,max(p.index)+1), fill_value=0.0)
        # Plot
        p.plot.bar(figsize=figsize, ax=ax)
        if savefig_loc is not None:
            plt.savefig(savefig_loc)
        return ax
    
    def qt_spa_by_spkage(self, spa_col:str = None, 
                        savefig_loc:str = None, figsize:Tuple[int,int] = (12,10), title:str = None):
        if (self.speaker_col) is None or (self.speaker_col not in self.data.columns):
            raise NameError("No valid speaker column")
        speakers = self.data[self.speaker_col].unique().tolist()
        if (len(speakers) == 1):
            fig, ax = plt.subplots(figsize=figsize)
            ax = self.qt_spa_over_age(self.data, spa_col=spa_col)
        else:
            fig, ax = plt.subplots(nrows=len(speakers), sharex=True, figsize=figsize)
            xlims = []
            for i, speaker in enumerate(speakers):
                self.qt_spa_over_age(self.data[self.data[self.speaker_col] == speaker], spa_col=spa_col, ax=ax[i])
                ax[i].set_title(f"Speaker: {speaker}")
                xlims.append(ax[i].get_xlim())
            # Setting custom xlim to avoid figure cutoff
            xlim_m = (  min(xlims, key=lambda x: x[0])[0], 
                        max(xlims, key=lambda x: x[1])[1] )
            plt.setp(ax, xlim=xlim_m)
            plt.setp(ax, xticks = range(int(xlim_m[0]), int(xlim_m[1])+1), 
                        xticklabels = range(int(xlim_m[0]), int(xlim_m[1])+1))
            
        if title is not None:
            fig.suptitle(title)
        if savefig_loc is not None:
            plt.savefig(savefig_loc)
        return (fig, ax)
    
    def prop_spa_over_age(self, data:pd.DataFrame = None, spa_col:str = None, 
                        savefig_loc:str = None, figsize:Tuple[int,int] = (10,5), ax=None):
        """Proportion of each tag / age group. Taking data as is, no filters.
        """
        if (self.tag_col is None) and (spa_col is None):
            raise ValueError("'tag_col' is not set for the data; 'spa_col' must be given when calling the function, currently None")
        elif (spa_col is None):
            print(f"Using '{self.tag_col}' as 'spa_col'.")
            spa_col = self.tag_col
        
        if data is None:
            print("Using class dataset as data (unfiltered).")
            data = self.data
        
        p = data.groupby(by=[self.age_col, spa_col]).agg({ spa_col: 'count' }).rename(columns={spa_col:'count'}
                ).reset_index().pivot_table(columns=self.age_col, index=spa_col, values='count').fillna(0)
        p = p/p.sum()
        # plotting
        p.plot.bar(figsize=figsize, ax=ax)
        if savefig_loc is not None:
            plt.savefig(savefig_loc)
    
    ### Study transitions
    def get_transitions(self, data:pd.DataFrame = None, spa_col:str = None, 
                        origin:str = None, second:str = None, 
                        remove_empty:bool = False, normalize:str = 'true') -> pd.DataFrame:
        """Compute theoretic transition table between tags.
        Data must be filtered (age of interest, removal of empty values) before being passed as argument.

        Input:
        -------
        remove_empty: `bool`
            | whether to include unlabeled sentences as "break" between to labels (which could lead to unnaturally occuring links)
        
        normalize: `str`
            | one of true|remove_empty|false
            | if 'true': divide by number of rows
            | if 'remove_empty': only count non empty tags
            | if 'false': no normalization
            
        """
        if (self.tag_col is None) and (spa_col is None):
            raise ValueError("'tag_col' is not set for the data; 'spa_col' must be given when calling the function, currently None")
        elif (spa_col is None):
            print(f"Using '{self.tag_col}' as 'spa_col'.")
            spa_col = self.tag_col
        
        if data is None:
            print("Using class dataset as data (unfiltered).")
            data = self.data

        dir_list = [None, 'CHI', 'MOT']
        if (origin not in dir_list) or (second not in dir_list):
            raise ValueError("Invalid argument; origin and second must be one of None|CHI|MOT")
        if normalize not in ['true', 'remove_empty', 'false']:
            raise ValueError(f"Invalid value for 'normalize': {normalize}; must be one of true|remove_empty|false")
        if remove_empty:
            df = data[~data[spa_col].isin(['NOL', 'NOT', 'NEE'])]
        else:
            df = data.copy(deep=True)
        # first: shift elements & remove 1rst line 
        df['prev_speaker'] = data[self.speaker_col].shift(1)
        df['prev_tag'] = data[spa_col].shift(1)
        df.dropna(subset=['prev_speaker', 'prev_tag'], inplace=True)
        # second: remove uninteresting rows
        if origin is not None:
            df = df[df['prev_speaker'] == origin]
        if second is not None:
            df = df[df[self.speaker_col] == second]
        # remove empty line - post shift
        # df = df[df[tag] != '']
        # groupby & pivot 
        n = df.shape[0]
        res = df.groupby(by=['prev_tag', spa_col]).agg({'sentence':'count'}).reset_index().pivot_table(values='sentence', index='prev_tag', columns=spa_col).fillna(0)
        # Check matrix is square
        res = fill_cm(res)
        # Normalize and return
        if normalize == 'true':
            return res/n
        elif normalize == 'remove_empty': # TODO
            return 
        else:
            return res
    

class PredAnalysis(SPAAnalysis):
    def __init__(self, model_report:str, model_metadata:str = None, **kwargs):
        super(PredAnalysis, self).__init__(data_path=model_report, **kwargs)
        # Checks
        expected_sheets = ['test_data', 'classification_report']
        if not (set(expected_sheets) <= set(self.sheetnames)):
            raise ValueError("Sheets '{}' missing from given report.".format("', '".join(list(set(expected_sheets) - set(self.sheetnames)))))
        # data is set
        self.data = self.dfs['test_data']
        self.classification_report = self.dfs['classification_report']

        # metadata
        self.metadata = {}
        if model_metadata is None:
            model_metadata = '/'.join(model_report.split('/')[:-1]) + '/metadata.txt'
        if not os.path.isfile(model_report) or 'metadata.txt' not in model_report:
            warnings.warn(f"Could not find any metadata at '{model_metadata}'; metadata will be empty.")
        else:
            text_file = open(model_metadata, "r")
            lines = text_file.readlines() # lines ending with "\n"
            for line in lines:
                arg_name, value = line[:-1].split(":\t")
                try:
                    self.metadata[arg_name] = ast.literal_eval(value)
                except ValueError as e:
                    if "malformed node or string" in str(e):
                        self.metadata[arg_name] = value
    

    ### Confusion matrix
    def confusion_matrix(self, speakers:Union[list, str] = None) -> pd.DataFrame:
        """Create a confusion matrix from the data; filters on speakers and age
        """
        if speakers is None:
            speakers = self.data[self.speaker_col].unique().tolist()
        elif isinstance(speakers, str):
            speakers = [speakers]
        return self.data[self.data[self.speaker_col].isin(speakers)].groupby(by=[self.truth_col]).count(self.pred_col)

    ### Histograms
    def qt_spa_true(self, savefig_loc:str = None, title:str = None):
        """Same as super but with the spa_col set as truth_col
        """
        return self.qt_spa_by_spkage(self.truth_col, savefig_loc, title)
    
    def qt_spa_pred(self, savefig_loc:str = None, title:str = None):
        """Same as super but with the spa_col set as truth_col
        """
        return self.qt_spa_by_spkage(self.pred_col, savefig_loc, title)

    ### Statistics
    def plot_scores(self, figsize:Tuple[int,int] = (18,5), 
                            plt_scale:str = 'linear', 
                            title:str = None, savefig_loc:str = None):
        """Plot how well a classifier fares depending on the amount of data for a given class; using 'classification_report' sheet
        """
        cols = ['precision','recall','f1-score'] # using 'support' for true count & proportions
        drop = ['accuracy', 'macro avg', 'weighted avg'] # useless rows
        # prepare data
        p = self.classification_report
        if set(drop) <= set(p.index):
            p.drop(drop, inplace=True)
        p.reset_index(inplace=True)
        p['percentage'] = p.support / p.support.sum()
        # Create graph
        fig, ax = plt.subplots(1,3, sharey=True, figsize=figsize)
        if plt_scale not in ["linear", "log", "symlog", "logit"]:
            raise ValueError("scale must be one of linear|log|symlog|logit")
        for axx in ax:
            axx.set_yscale(plt_scale)
            axx.set_xscale(plt_scale)

        for i, col in enumerate(cols):
            ax[i].grid(True, alpha=0.2, ls='-')
            ax[i].scatter(p['percentage'], p[col])
            for row in p.iterrows():
                ax[i].annotate(row[1]['index'], xy=(row[1]['percentage'], row[1][col]+0.01))
            ax[i].set_title(col)
        if title is not None:
            fig.suptitle(title)
        
        # save or return
        if savefig_loc is not None:
            plt.savefig(savefig_loc)
        return (fig, ax)
    
    def plot_distribution(self, figsize:Tuple[int,int] = (10,10), 
                            plt_scale:str = 'linear', 
                            title:str = None, savefig_loc:str = None):
        """Plot distribution of tags truth vs predicted
        """
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.plot([0,1], [0,1], alpha=0.2) # TODO: make sure the line is curved when logscale
        # Options
        if plt_scale not in ["linear", "log", "symlog", "logit"]:
            raise ValueError("scale must be one of None|linear|log|symlog|logit")
        ax.set_yscale(plt_scale)
        ax.set_xscale(plt_scale)
        # Data
        counts = pd.concat([self.data[self.truth_col].value_counts(), self.data[self.pred_col].value_counts()], axis=1).fillna(0)/self.data.shape[0] # creating proportions
        ax.scatter(counts[self.truth_col], counts[self.pred_col])
        # Labels on plot
        for row in counts.reset_index().iterrows():
            ax.annotate(row[1]['index'], xy=(row[1][self.truth_col], row[1][self.pred_col]))
        # Settings
        ax.set_title(f'{self.truth_col} vs {self.pred_col}')
        ax.set_xlim(-0.005,counts[self.truth_col].max()+0.005)
        ax.set_ylim(-0.005,counts[self.pred_col].max()+0.005)
        ax.set_xlabel(f'{self.truth_col}: label in data (percentage)')
        ax.set_ylabel(f'{self.pred_col}: label in data (percentage)')
        # Return
        if savefig_loc is not None:
            plt.savefig(savefig_loc)
        return (fig, ax)
    

    ### Consistency
    def consistency_check(self, speaker:str = None, styler:bool=False):
        """Checking whether each child / parent is as easily predictable as the others
        """
        if self.child_col is None:
            raise ValueError("'child_col' must be set to check consistency.")
        if isinstance(speaker, str):
            speaker = [speaker]
        else: 
            speaker = ['CHI', 'MOT']

        cc = {} # results
        for c_name in self.data[self.child_col].unique().tolist(): # for each child
            cc[c_name] = { 'overall': accuracy_score(self.data[self.data[self.child_col] == c_name].y_true, 
                                        self.data[self.data[self.child_col] == c_name].y_pred, normalize = True) 
                        }
            if self.age_col is not None: # for each age
                for age in sorted(self.data[self.data[self.child_col] == c_name][self.age_col].unique().tolist()):
                    cc[c_name][age] = accuracy_score(
                                self.data[(self.data[self.child_col] == c_name) & (self.data[self.age_col] == age) & (self.data[self.speaker_col].isin(speaker))].y_true, 
                                self.data[(self.data[self.child_col] == c_name) & (self.data[self.age_col] == age) & (self.data[self.speaker_col].isin(speaker))].y_pred, 
                                normalize = True
                    )
        # Style and return
        cc = pd.DataFrame(cc).T
        if styler: # to be used in excel
            return (cc.style
                        .format("{:.2%}")
                        .background_gradient(cmap='coolwarm')
                    )
        return cc


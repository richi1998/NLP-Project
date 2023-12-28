import random
from collections import defaultdict, Counter    
import gensim
from gensim.models import KeyedVectors
import gensim.downloader
import sys
import time
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from sklearn.model_selection import train_test_split

def read_data_from_dataframe(df):
    # Group by 'Sentence #' and create lists of words and tags
    grouped = df.groupby('Sentence #')
    X = grouped['Word'].apply(list).tolist()
    Y = grouped['Tag'].apply(list).tolist()
    return X, Y
    

PAD = '___PAD___'
UNKNOWN = '___UNKNOWN___'
BOS = '___BOS___'
EOS = '___EOS___'

class Vocabulary:
    """Manages the numerical encoding of the vocabulary."""
    
    def __init__(self, max_voc_size=None, include_unknown=True, lower=False,
                 character=False, gensim_model=None):

        self.include_unknown = include_unknown
        self.dummies = [PAD, UNKNOWN, BOS, EOS] if self.include_unknown else [PAD, BOS, EOS]
        
        self.character = character
        
        if not gensim_model:
            # String-to-integer mapping
            self.stoi = None
            # Integer-to-string mapping
            self.itos = None
            # Maximally allowed vocabulary size.
            self.max_voc_size = max_voc_size
            self.lower = lower
            self.vectors = None
        else:
            self.vectors = gensim_model[0]
            self.itos = self.dummies + gensim_model[1]
            self.stoi = {s:i for i, s in enumerate(self.itos)}
            self.lower = not gensim_model[2]
            
    def make_embedding_layer(self, finetune=True, emb_dim=None):
        if self.vectors is not None:
            emb_dim = self.vectors.shape[1]
            emb_layer = nn.Embedding(len(self.itos), emb_dim, sparse=False)

            with torch.no_grad():
                # Copy the pre-trained embedding weights into our embedding layer.
                emb_layer.weight[len(self.dummies):, :] = self.vectors

            #print(f'Emb shape: {emb_layer.weight.shape}, voc size: {len(self.itos)}')
        else:
            emb_layer = nn.Embedding(len(self.itos), emb_dim, sparse=False)
        
        if not finetune:
            # If we don't fine-tune, create a tensor where we don't compute the gradients.
            emb_layer.weight = nn.Parameter(emb_layer.weight, requires_grad=False)
        
        return emb_layer
        
    def build(self, seqs):
        """Builds the vocabulary."""
        
        if self.character:
            seqs = [ [c for w in seq for c in w] for seq in seqs ]
        
        if self.lower:
            seqs = [ [s.lower() for s in seq] for seq in seqs ]
        
        # Sort all words by frequency
        word_freqs = Counter(w for seq in seqs for w in seq)
        word_freqs = sorted(((f, w) for w, f in word_freqs.items()), reverse=True)

        # Build the integer-to-string mapping. The vocabulary starts with the two dummy symbols,
        # and then all words, sorted by frequency. Optionally, limit the vocabulary size.
        
        if self.max_voc_size:
            self.itos = self.dummies + [ w for _, w in word_freqs[:self.max_voc_size-len(dummies)] ]
        else:
            self.itos = self.dummies + [ w for _, w in word_freqs ]

        # Build the string-to-integer map by just inverting the aforementioned map.
        self.stoi = { w: i for i, w in enumerate(self.itos) }
                
    def encode(self, seqs):
        """Encodes a set of documents."""
        unk = self.stoi.get(UNKNOWN)
        bos = self.stoi.get(BOS)
        eos = self.stoi.get(EOS)
        
        if self.character:
            if self.lower:
                seqs = [ [[c for c in w.lower()] for w in seq] for seq in seqs ]
            return [[[bos,eos]]+[[bos]+[self.stoi.get(c, unk) for c in w]+[eos] for w in seq]+[[bos,eos]] for seq in seqs]
        else:
            if self.lower:
                seqs = [ [s.lower() for s in seq] for seq in seqs ]
            return [[bos]+[self.stoi.get(w, unk) for w in seq]+[eos] for seq in seqs]

    def get_unknown_idx(self):
        """Returns the integer index of the special dummy word representing unknown words."""
        return self.stoi[UNKNOWN]
    
    def get_pad_idx(self):
        """Returns the integer index of the special padding dummy word."""
        return self.stoi[PAD]
    
    def __len__(self):
        return len(self.itos)
    
    

class SequenceDataset(Dataset):
    """A Dataset that stores a list of sentences and their corresponding labels."""
    def __init__(self, Xwords, Y, Xchars=None, word_dropout_prob=None, word_dropout_id=None):
        self.Xwords = Xwords
        self.Xchars = Xchars
        self.Y = Y
        self.word_dropout_prob = word_dropout_prob
        self.word_dropout_id = word_dropout_id
        
    def __getitem__(self, idx):
        if self.word_dropout_prob:
            words = [ w if random.random() > self.word_dropout_prob else self.word_dropout_id for w in self.Xwords[idx] ]
        else:
            words = self.Xwords[idx]
        
        if self.Xchars:
            return words, self.Y[idx], self.Xchars[idx]
        else:
            return words, self.Y[idx]
        
    def __len__(self):
        return len(self.Y)

class SequenceBatcher:
    """A collator that builds a batch from a number of sentence--labeling pairs."""
    
    def __init__(self, device):
        self.device = device
    
    def __call__(self, XY):
        """Build a batch from a number of sentences. Returns two tensors X and Y, where
        X is the sentence tensor, of shape [n_sentences, max_sen_length]

        and 
        
        Y is the label tensor, of the same shape as X.
        """
        
        # Assume that the padding id is 0.
        pad_id = 0
                
        # How long is the longest document in this batch?
        max_sen_len = max(len(row[0]) for row in XY)

        # Build the document tensor. We pad the shorter documents so that all documents
        # have the same length.
        
        Xpadded = torch.as_tensor([row[0] + [pad_id]*(max_sen_len-len(row[0])) for row in XY], device=self.device)
        
        # Build the label tensor.
        Ypadded = torch.as_tensor([row[1] + [pad_id]*(max_sen_len-len(row[1])) for row in XY], device=self.device)

        if len(XY[0]) == 2:
            return Xpadded, None, Ypadded
        else:
            max_word_len = max(len(w) for _, _, xc in XY for w in xc)
            Xcpadded = [xc + [[]]*(max_sen_len-len(xc)) for _, _, xc in XY]
            Xcpadded = [[w + [pad_id]*(max_word_len-len(w)) for w in xc] for xc in Xcpadded]
            Xcpadded = torch.as_tensor(Xcpadded, device=self.device)            
            return Xpadded, Xcpadded, Ypadded
       
        


def load_gensim_vectors(model_file, builtin=False, limit=None):
    using_new_gensim = gensim.__version__[0] == '4'

    print(f"Loading model '{model_file}' via gensim...", end='')
    sys.stdout.flush()
    if builtin:
        gensim_model = gensim.downloader.load(model_file)
    else:
        gensim_model = KeyedVectors.load_word2vec_format(model_file, binary=True, limit=limit)
    if not limit:
        if using_new_gensim:
            limit = len(gensim_model.index_to_key)
        else:
            limit = len(gensim_model.index2word)
    vectors = torch.FloatTensor(gensim_model.vectors[:limit])
    
    if using_new_gensim:
        voc = gensim_model.index_to_key[:limit]
    else:
        voc = gensim_model.index2word[:limit]
    
    is_cased = False
    for w in voc:
        w0 = w[0]
        if w0.isupper():
            is_cased = True
            break
    
    print(' done!')
    return vectors, voc, is_cased


# Aligns encoded output BIO labels with the word pieces created
# by a BERT tokenizer.
def remap_entity_indices(word_tokenized, label_encoded, voc):
    out = []
    O = voc.stoi['O']
    for i, y in enumerate(label_encoded):
        tokens_sen = word_tokenized[i]
        new_len = len(tokens_sen.ids)
        spans = to_spans(y, voc)
        y_new = [y[0]]
        prev = 1
        for start, (lbl, end) in sorted(spans.items()):
            start_remapped = tokens_sen.word_to_tokens(start-1)#[0]
            if start_remapped is None:
                continue
            y_new.extend([O]*(start_remapped[0]-prev))
            y_new.append(voc.stoi['B-' + lbl])
                        
            end_remapped = tokens_sen.word_to_tokens(end-1)
            if end_remapped is None:
                end_remapped = (new_len-1, )
            
            y_new.extend([voc.stoi['I-' + lbl]]*(end_remapped[0]-start_remapped[0]-1))
            prev = end_remapped[0]
        y_new.extend([O]*(new_len-prev-1))
        y_new.append(y[-1])
        out.append(y_new)
    return out


# Convert a list of BIO labels, coded as integers, into spans identified by a beginning, an end, and a label.
# To allow easy comparison later, we store them in a dictionary indexed by the start position.
def to_spans(l_ids, vocab):
    spans = {}
    current_lbl = None
    current_start = None
    for i, l_id in enumerate(l_ids):
        l = vocab.itos[l_id]

        if l[0] == 'B': 
            # Beginning of a named entity: B-something.
            if current_lbl:
                # If we're working on an entity, close it.
                spans[current_start] = (current_lbl, i)
            # Create a new entity that starts here.
            current_lbl = l[2:]
            current_start = i
        elif l[0] == 'I':
            # Continuation of an entity: I-something.
            if current_lbl:
                # If we have an open entity, but its label does not
                # correspond to the predicted I-tag, then we close
                # the open entity and create a new one.
                if current_lbl != l[2:]:
                    spans[current_start] = (current_lbl, i)
                    current_lbl = l[2:]
                    current_start = i
            else:
                # If we don't have an open entity but predict an I tag,
                # we create a new entity starting here even though we're
                # not following the format strictly.
                current_lbl = l[2:]
                current_start = i
        else:
            # Outside: O.
            if current_lbl:
                # If we have an open entity, we close it.
                spans[current_start] = (current_lbl, i)
                current_lbl = None
                current_start = None
    return spans

# Compares two sets of spans and records the results for future aggregation.
def compare(gold, pred, stats):
    for start, (lbl, end) in gold.items():
        stats['total']['gold'] += 1
        stats[lbl]['gold'] += 1
    for start, (lbl, end) in pred.items():
        stats['total']['pred'] += 1
        stats[lbl]['pred'] += 1
    for start, (glbl, gend) in gold.items():
        if start in pred:
            plbl, pend = pred[start]
            if glbl == plbl and gend == pend:
                stats['total']['corr'] += 1
                stats[glbl]['corr'] += 1

# This function combines the auxiliary functions we defined above.
def evaluate_iob(words, predicted, gold, vocab, stats):
    
    pad_id = vocab.get_pad_idx()
    padding = list((words == pad_id).reshape(-1).cpu().numpy())
                
    # The gold-standard labels are assumed to be an integer tensor of shape
    # (n_sentences, max_len).
    gold_cpu = gold.cpu().numpy()
    gold_cpu = list(gold_cpu.reshape(-1))

    if not isinstance(predicted, list):        
        pred_flat = predicted.reshape(-1).cpu().numpy()
    else:
        pred_flat = [l for sen in predicted for l in sen]
    pred_flat = [pad_id if is_pad else l for l, is_pad in zip(pred_flat, padding)]
    
    # Compute spans for the gold standard and prediction.
    gold_spans = to_spans(gold_cpu, vocab)
    pred_spans = to_spans(pred_flat, vocab)

    # Finally, update the counts for correct, predicted and gold-standard spans.
    compare(gold_spans, pred_spans, stats)

# Computes precision, recall and F-score, given a dictionary that contains
# the counts of correct, predicted and gold-standard items.
def prf(stats):
    if stats['pred'] == 0:
        return 0, 0, 0
    p = stats['corr']/stats['pred']
    r = stats['corr']/stats['gold']
    if p > 0 and r > 0:
        f = 2*p*r/(p+r)
    else:
        f = 0
    return p, r, f

from IPython.core.display import display, HTML

def show_entities(tagger, sentences):
    
    tagged_sentences = tagger.predict(sentences)
    if tagger.bert_tokenizer is not None:
        word_encoded = tagger.bert_tokenizer(sentences, is_split_into_words=True, truncation=True, 
                                             max_length=tagger.params.bert_max_len).input_ids
        sentences = [[tagger.bert_tokenizer.decode(i) for i in s[1:-1]] for s in word_encoded]
    
    styles = {
        'LOC': 'background-color: #aaffaa; color: black;',
        'PER': 'background-color: #aaaaff; color: black;',
        'ORG': 'background-color: #ff8800; color: black;',
        'MISC': 'background-color: #00ffff; color: black;',

        'Disease': 'background-color: #ff3333; color: white;',
        'Chemical': 'background-color: #44bbff; color: white;'
    }
    content = ['<div style="font-size:150%; line-height: 150%;">']

    for tokens, tags in zip(sentences, tagged_sentences):
        content.append('<div>')
        current_entity = None
        for token, tag in zip(tokens, tags):
            if tag[0] not in ['B', 'I']:
                if current_entity:
                    content.append('</b>')
                    current_entity = None
                content.append(' ')
            elif tag[0] == 'B':
                if current_entity:
                    content.append('</b>')
                content.append(' ')
                current_entity = tag[2:]
                content.append(f'<b style="{styles[current_entity]} border-radius: 3px; padding: 3px;">')
                content.append(f'<sup style=font-size:small;><tt>{current_entity}</tt></sup> ')

            else:
                entity = tag[2:]
                if entity == current_entity:
                    content.append(' ')
                elif current_entity is None:
                    content.append(' ')
                    content.append('<sup style=font-size:small;><tt>[ERROR]</tt></sup> ')
                    content.append(f'<b style="{styles[entity]} border-radius: 3px; padding: 3px;">')
                    content.append(f'<sup style=font-size:small;><tt>{entity}</tt></sup> ')
                else:
                    content.append('</b>')
                    content.append(' ')
                    content.append('<sup style=font-size:small;><tt>[ERROR]</tt></sup> ')
                    content.append(f'<b style="{styles[entity]} border-radius: 3px; padding: 3px;">')
                    content.append(f'<sup style=font-size:small;><tt>{entity}</tt></sup> ')
                current_entity = entity
            content.append(token)
        if current_entity:
            content.append('</b>')
        content.append('</div>')
    content.append('</div>')    
    html = ''.join(content).strip()
    display(HTML(html))
        
        
class SequenceLabeler:
    
    def __init__(self, params, model_factory, pretrained_word_emb=None, bert_tokenizer=None):
        self.params = params        
        self.model_factory = model_factory
        self.pretrained_word_emb = pretrained_word_emb
        self.bert_tokenizer = bert_tokenizer
        self.verbose = bert_tokenizer is not None
            
    def preprocess(self, Xtrain, Ytrain, Xval, Yval):
        p = self.params
        
        self.label_voc = Vocabulary(include_unknown=False)
        self.label_voc.build(Ytrain)
        self.n_labels = len(self.label_voc)
        train_lbl_encoded = self.label_voc.encode(Ytrain)
        val_lbl_encoded = self.label_voc.encode(Yval)
        
        if not self.bert_tokenizer:
            # If we are using a pre-trained word embedding model, then we use its built-in vocabulary;
            # otherwise, we build the word vocabulary from the data.

            self.word_voc = Vocabulary(include_unknown=True, gensim_model=self.pretrained_word_emb)
            if not self.pretrained_word_emb:
                self.word_voc.build(Xtrain)            

            train_word_encoded = self.word_voc.encode(Xtrain)
            val_word_encoded = self.word_voc.encode(Xval)
            
            if p.use_characters:
                self.char_voc = Vocabulary(include_unknown=True, character=True)            
                self.char_voc.build(Xtrain)
                train_char_encoded = self.char_voc.encode(Xtrain)
                val_char_encoded = self.char_voc.encode(Xval)
            else:                
                self.char_voc = None
                train_char_encoded = None
                val_char_encoded = None
            dropout_id = self.word_voc.get_unknown_idx()
        else:
            train_tokenized = self.bert_tokenizer(Xtrain, is_split_into_words=True, truncation=True, max_length=p.bert_max_len)
            val_tokenized = self.bert_tokenizer(Xval, is_split_into_words=True, truncation=True, max_length=p.bert_max_len)
            train_lbl_encoded = remap_entity_indices(train_tokenized, train_lbl_encoded, self.label_voc)
            val_lbl_encoded = remap_entity_indices(val_tokenized, val_lbl_encoded, self.label_voc)
            train_word_encoded = train_tokenized.input_ids
            val_word_encoded = val_tokenized.input_ids
            train_char_encoded = None
            val_char_encoded = None
            dropout_id = self.bert_tokenizer.unk_token_id

        # Put the training and validation data into Datasets and DataLoaders for managing minibatches.
        self.batcher = SequenceBatcher(p.device)        
        train_dataset = SequenceDataset(train_word_encoded, train_lbl_encoded, train_char_encoded,
                                        word_dropout_prob=p.word_dropout_prob, word_dropout_id=dropout_id)
        self.train_loader = DataLoader(train_dataset, p.batch_size, shuffle=True, collate_fn=self.batcher)        
        val_dataset = SequenceDataset(val_word_encoded, val_lbl_encoded, val_char_encoded)
        self.val_loader = DataLoader(val_dataset, p.batch_size, shuffle=False, collate_fn=self.batcher)
            
    def fit(self, Xtrain, Ytrain, Xval, Yval):
        
        p = self.params
        
        # Setting a fixed seed for reproducibility.
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)
                           
        # Preprocess the data, build vocabularies and data loaders.
        self.preprocess(Xtrain, Ytrain, Xval, Yval)
            
        # Now, let's build the model!
        self.model = self.model_factory(self)

        # Create the sequence labeling neural network defined above.
        
        self.model.to(p.device)
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=p.learning_rate, weight_decay=p.weight_decay)
        #optimizer = torch.optim.SparseAdam(self.model.parameters(), 
        #                                   lr=p.learning_rate)

        # Cross-entropy loss function that we will use to optimize the model.
        # In particular, note that by using ignore_index, we will not compute the loss 
        # for the positions where we have a padding token.
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.label_voc.get_pad_idx())
        
        self.history = defaultdict(list)
        best_f1 = -1        
            
        for i in range(p.n_epochs):

            t0 = time.time()

            loss_sum = 0

            self.model.train()
            for j, batch in enumerate(self.train_loader, 1):
                
                # Compute the output scores.                
                if p.use_characters:
                    scores = self.model(batch[0], batch[1])
                else:
                    scores = self.model(batch[0])                
                
                # The scores tensor has the shape (n_sentences, n_words, n_labels).
                # We reshape this to (n_sentences*n_words, n_labels) because the loss requires
                # a 2-dimensional tensor. Similarly for the gold standard label tensor.                
                loss = loss_func(scores.view(-1, len(self.label_voc)), batch[2].view(-1))
                    
                optimizer.zero_grad()            
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                
                if self.verbose:
                    print('.', end='')
                    sys.stdout.flush()
                    if j % 50 == 0:
                        print(f' ({j})')               
            if self.verbose:
                print()
                
            train_loss = loss_sum / len(self.train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Evaluate on the validation set.
            stats = defaultdict(Counter)

            self.model.eval()
            with torch.no_grad():
                for j, batch in enumerate(self.val_loader, 1):
                    
                    if p.use_characters:
                        scores = self.model(batch[0], batch[1])
                    else:
                        scores = self.model(batch[0])   
                    
                    # Compute the highest-scoring labels at each word position.
                    predicted = scores.argmax(dim=2)
                    
                    # Update the evaluation statistics for this batch.
                    evaluate_iob(batch[0], predicted, batch[2], self.label_voc, stats)
                    
                    if self.verbose:
                        print('.', end='')
                        sys.stdout.flush()
                        if j % 50 == 0:
                            print(f' ({j})')
            if self.verbose:
                print()
                        
            # Compute the overall F-score for the validation set.
            _, _, val_f1 = prf(stats['total'])

            self.history['val_f1'].append(val_f1)
            if val_f1 > best_f1:
                best_f1 = val_f1

            t1 = time.time()
            print(f'Epoch {i+1}: train loss = {train_loss:.4f}, val f1: {val_f1:.4f}, time = {t1-t0:.4f}')
           
        # After the final evaluation, we print more detailed evaluation statistics, including
        # precision, recall, and F-scores for the different types of named entities.
        print()
        print('Final evaluation on the validation set:')
        p, r, f1 = prf(stats['total'])
        print(f'Overall: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')
        for label in stats:
            if label != 'total':
                p, r, f1 = prf(stats[label])
                print(f'{label:4s}: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')
        
        self.stats = stats
        return best_f1
        
    def predict(self, sentences):
        # This method applies the trained model to a list of sentences.
                    
        if not self.bert_tokenizer:
            word_encoded = self.word_voc.encode(sentences)            
            if self.params.use_characters:
                char_encoded = self.char_voc.encode(sentences)
            else:                
                char_encoded = None
        else:
            word_encoded = self.bert_tokenizer(sentences, is_split_into_words=True, truncation=True, 
                                               max_length=self.params.bert_max_len).input_ids
            char_encoded = None
            
        Ydummy = [[0]*len(x) for x in word_encoded]
                
        dataset = SequenceDataset(word_encoded,
                                  Ydummy,
                                  char_encoded)
        loader = DataLoader(dataset, self.params.batch_size, shuffle=False, collate_fn=self.batcher)
                
        out = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                if self.params.use_characters:
                    scores = self.model(batch[0], batch[1])
                else:
                    scores = self.model(batch[0])                
                predicted = scores.argmax(dim=2) 
                
                # Convert the integer-encoded tags to tag strings.
                for pred_sen in predicted.cpu().numpy():
                    tokens = word_encoded[len(out)]
                    out.append([self.label_voc.itos[pred_id] for _, pred_id in zip(tokens, pred_sen[1:-1])])
        return out
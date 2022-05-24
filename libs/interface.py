import tkinter
from tkinter import *
 
UPDATE_RATE = 1000
 
class Application(tkinter.Frame):
    """ GUI """
 
    def __init__(self, master):
        """ Initialize the Frame"""
        tkinter.Frame.__init__(self, master)
        self.abc="ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
        self.contador=0
        self.grid()
        self.create_widgets()
        
    
    def run(self):
      self.updater()

    def set_Function_Loop(self,functionLoop):
        self.functionLoop = functionLoop

    def create_widgets(self):
        """Create button. """
#         import os # import at top
#         import subprocess # doing nothing
        # Router
        self.button1 = tkinter.Button(self)
        self.button1.pack(anchor=CENTER)
        self.button1.config(
            fg="blue",  
            font=("Verdana",80)
        )
        self.button1.grid(row=0, column=10, rowspan=1, columnspan=2)
        self.button1["text"] = ""        
    
 
    def update_button1(self):
        # Ping
        self.functionLoop(self)        
 
    def updater(self):
        self.update_button1()
        self.after(1000, self.updater)
 



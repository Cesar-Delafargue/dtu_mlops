def test_data():
      
       from data import mnist
      
       assert mnist()[0].shape[0] == 40000 
       assert mnist()[1].shape[0] == 5000
      
      #assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format



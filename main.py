import importlib

import pipeline.eda01
import pipeline.preprocess02
import pipeline.baseline03
import pipeline.evaluation04
import pipeline.predict05

def main():
  pipeline.eda01.main()
  pipeline.preprocess02.main()
  pipeline.baseline03.main()
  pipeline.evaluation04.main()
  pipeline.predict05.main()

if __name__ == "__main__":
  main()
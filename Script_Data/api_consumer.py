#!/usr/bin/env python3
import requests
import os
import logging

log = logging.getLogger("logger")
log.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',datefmt='%Y/%m/%d %H:%M:%S')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

log.addHandler(console_handler)


class Consumer() :
    def __init__(self, token):
        self.target_url = "http://204.235.60.194/exrxapi/v1/allinclusive/exercises"
        self.exercise_list = ["back"]
        self.headers ={ "Authorization" : "Bearer {}".format(token) }

        self.storage = "../data/"
        if not os.path.exists(self.storage):
            os.makedirs(self.storage)

    def _get_exercises(self, exercisename):

        payload={
                "exercisename" : exercisename
                }

        response = requests.get(self.target_url, headers = self.headers, params = payload)
        self.process_response(response.json())

    def process_response(self, response) :
        for i in range(30) :
            if response.get("exercises", None) is not None :
                if response["exercises"].get(str(i+1), None) is not None :
                    gif_url = "http:{}".format( response["exercises"][str(i+1)].get("GIF_Img") )
                    exercise_name = response["exercises"][str(i+1)].get("Exercise_Name").replace(" ", "_")

                    self.download_gif(gif_url, exercise_name)


    def download_gif(self, gif_url, exercise_name) :
        exercise_path = "{}{}/".format(self.storage, exercise_name)
        if not os.path.exists(exercise_path):
            os.makedirs(exercise_path)

        gif_name = gif_url.split("/")[-1:][0]
        gif_path = "{}{}".format(exercise_path, gif_name)

        log.info("Downloading GIF {} to {}".format(gif_url, gif_path))

        dl_response = requests.get(gif_url)

        with open(gif_path, "wb") as _buffer :
            _buffer.write(dl_response.content)

    def main_worker(self) :
        for exercise_name in self.exercise_list :

            log.info("-------------------------------")
            log.info("Request for exercisename query param : {}".format(exercise_name))
            self._get_exercises(exercise_name)

if __name__=="__main__":

    token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOlwvXC8yMDQuMjM1LjYwLjE5NFwvZnVzaW9cL3B1YmxpY1wvaW5kZXgucGhwIiwic3ViIjoiMjIyNDdmY2MtMGI5Mi01MmFlLTg0NGMtY2FjOWQ3MDQ3YjZlIiwiaWF0IjoxNjMwMzA1NTcxLCJleHAiOjE2MzAzMDkxNzEsIm5hbWUiOiJsZXdhZ29uZXhyeCJ9.5IuRmqVuygY_nbTyHQtRVbfRiyZprcTqXI2ppcNW6Gg"
    c = Consumer(token)
    c.main_worker()

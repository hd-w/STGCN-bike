import csv
import codecs
import urllib.request
import sys

# This is the core of our weather query URL
BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/'

# Set up the location parameter for our query
QueryLocation = '&location=' + urllib.parse.quote(sys.argv[1])

# Set up the query type parameter for our query ('FORECAST' or 'HISTORY')
QueryType=sys.argv[2].upper()

# Set up the key parameter for our query
QueryKey = '&key=NDXJ94SQ3ZQLKXSM4USZ52E77'

# Set up the date parameters for our query. Used only for historical weather data requests
FromDateParam = sys.argv[4]
ToDateParam = sys.argv[5]

# Set up the specific parameters based on the type of query
if QueryType == 'FORECAST':
    print(' - Fetching forecast data')
    QueryTypeParams = 'forecast?&aggregateHours=24&unitGroup=us&shortColumnNames=false'
else:
    print(' - Fetching history for date: ', FromDateParam,'-',ToDateParam)

    # History requests require a date.  We use the same date for start and end since we only want to query a single date in this example
    QueryDate = '&startDateTime=' + FromDateParam + 'T00:00:00&endDateTime=' +ToDateParam + 'T00:00:00'
    QueryTypeParams = 'history?&aggregateHours=24&unitGroup=us&dayStartTime=0:0:00&dayEndTime=0:0:00' + QueryDate


# Build the entire query
URL = BaseURL + QueryTypeParams + QueryLocation + QueryKey

# Build the entire query
URL = BaseURL + QueryTypeParams + QueryLocation + QueryKey

print(' - Running query URL: ', URL)
print()

# Parse the results as CSV
CSVBytes = urllib.request.urlopen(URL)
CSVText = csv.reader(codecs.iterdecode(CSVBytes, 'utf-8'))
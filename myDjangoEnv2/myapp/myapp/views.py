from django.http import HttpResponse
from django.shortcuts import render
import pickle
from sklearn.ensemble import RandomForestClassifier


def home(request):
    return render(request, "home.html")

def result(request):
    if request.method == 'POST':
        a=[]
        try:
            # Retrieve the values from the form
            distance_from_home = float(request.POST.get('distance_from_home'))
            distance_from_last_transaction = float(request.POST.get('distance_from_last_transaction'))
            ratio_to_median_purchase_price = float(request.POST.get('ratio_to_median_purchase_price'))
            repeat_retailer = int(request.POST.get('repeat_retailer'))
            used_chip = int(request.POST.get('used_chip'))
            used_pin_number = int(request.POST.get('used_pin_number'))
            online_order = int(request.POST.get('online_order'))
            a.extend([distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,
                      repeat_retailer,used_chip,used_pin_number,online_order])
            #loading model
            model=pickle.load(open("best_model.pkl",'rb'))
            result=""
            fraud_prediction = model.predict([a])
            print(fraud_prediction[0])
            if fraud_prediction[0]==1:
                result="Transaction is Fraudulent"
            else:
                result="Normal Transaction"
            return render(request, 'result.html', {'result': result})

        except ValueError:
            return ValueError
    else:
        return HttpResponse("Invalid request method.", status=405)